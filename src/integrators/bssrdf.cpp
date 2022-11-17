#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/records.h>
#include <random>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class BSSRDFIntegrator : public MonteCarloIntegrator<Float, Spectrum> {

public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth,
                   m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Emitter, EmitterPtr, BSDF, BSDFPtr, Medium,
                    MediumPtr, PhaseFunctionContext)
    BSSRDFIntegrator(const Properties &props) : Base(props) {}

    MI_INLINE
    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active) const override {
        // If there is an environment emitter and emitters are visible: all rays
        // will be valid Otherwise, it will depend on whether a valid
        // interaction is sampled
        Mask valid_ray =
            !m_hide_emitters && dr::neq(scene->environment(), nullptr);

        // For now, don't use ray differentials
        Ray3f ray = ray_;
        // Tracks radiance scaling due to index of refraction changes
        Float eta    = 1.f;
        UInt32 depth = 0;
        Spectrum throughput(1.f), result(0.f);

        // Variables caching information from the previous bounce
        Interaction3f prev_si = dr::zeros<Interaction3f>();
        Float prev_bsdf_pdf   = 1.f;
        Bool prev_bsdf_delta  = true;
        BSDFContext bsdf_ctx;
        dr::Loop<Bool> loop("BSSRDF", sampler, ray, throughput, result, eta,
                            depth, valid_ray, prev_si, prev_bsdf_pdf,
                            prev_bsdf_delta, active);

        loop.set_max_iterations(m_max_depth);

        while (loop(active)) {
            SurfaceInteraction3f si =
                scene->ray_intersect(ray,
                                     /* ray_flags = */ +RayFlags::All,
                                     /* coherent = */ dr::eq(depth, 0u));
            // ---------------------- Direct emission ----------------------

            /* dr::any_or() checks for active entries in the provided boolean
               array. JIT/Megakernel modes can't do this test efficiently as
               each Monte Carlo sample runs independently. In this case,
               dr::any_or<..>() returns the template argument (true) which means
               that the 'if' statement is always conservatively taken. */
            if (dr::any_or<true>(dr::neq(si.emitter(scene), nullptr))) {
                DirectionSample3f ds(scene, si, prev_si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!prev_bsdf_delta))
                    em_pdf = scene->pdf_emitter_direction(prev_si, ds,
                                                          !prev_bsdf_delta);

                // Compute MIS weight for emitter sample from previous bounce
                Float mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf);

                // Accumulate, being careful with polarization (see spec_fma)
                result = spec_fma(throughput,
                                  ds.emitter->eval(si, prev_bsdf_pdf > 0.f) *
                                      mis_bsdf,
                                  result);
            }

            // Continue tracing the path at this point?
            Bool active_next = (depth + 1 < m_max_depth) && si.is_valid();

            if (dr::none_or<false>(active_next))
                break; // early exit for scalar mode

            // ---------------------- Emitter sampling ----------------------

            // Perform emitter sampling?
            BSDFPtr bsdf = si.bsdf(ray);
            Mask active_em =
                active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            if (dr::any_or<true>(active_em)) {
                // Sample the emitter
                auto [ds, em_weight] = scene->sample_emitter_direction(
                    si, sampler->next_2d(), true, active_em);
                active_em &= dr::neq(ds.pdf, 0.f);

                /* Given the detached emitter sample, recompute its contribution
                   with AD to enable light source optimization. */
                if (dr::grad_enabled(si.p)) {
                    ds.d = dr::normalize(ds.p - si.p);
                    Spectrum em_val =
                        scene->eval_emitter_direction(si, ds, active_em);
                    em_weight =
                        dr::select(dr::neq(ds.pdf, 0), em_val / ds.pdf, 0);
                }

                // Evaluate BSDF * cos(theta)
                Vector3f wo = si.to_local(ds.d);
                auto [bsdf_val, bsdf_pdf] =
                    bsdf->eval_pdf(bsdf_ctx, si, wo, active_em);
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                // Accumulate, being careful with polarization (see spec_fma)
                result[active_em] =
                    spec_fma(throughput, bsdf_val * em_weight * mis_em, result);
            }

            // ---------------------- BSDF sampling ----------------------

            Float sample_1   = sampler->next_1d();
            Point2f sample_2 = sampler->next_2d();

            auto [bsdf_sample, bsdf_weight] =
                bsdf->sample(bsdf_ctx, si, sample_1, sample_2, active_next);
            bsdf_weight =
                si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

            /* When the path tracer is differentiated, we must be careful that
               the generated Monte Carlo samples are detached (i.e. don't track
               derivatives) to avoid bias resulting from the combination of
               moving samples and discontinuous visibility. We need to
               re-evaluate the BSDF differentiably with the detached sample in
               that case. */
            if (dr::grad_enabled(ray)) {
                ray = dr::detach<true>(ray);

                // Recompute 'wo' to propagate derivatives to cosine term
                Vector3f wo = si.to_local(ray.d);
                auto [bsdf_val, bsdf_pdf] =
                    bsdf->eval_pdf(bsdf_ctx, si, wo, active);
                bsdf_weight[bsdf_pdf > 0.f] = bsdf_val / dr::detach(bsdf_pdf);
            }

            // ------ Update loop variables based on current interaction ------

            throughput *= bsdf_weight;
            eta *= bsdf_sample.eta;
            valid_ray |= active && si.is_valid() &&
                         !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

            // Information about the current vertex needed by the next iteration
            prev_si       = si;
            prev_bsdf_pdf = bsdf_sample.pdf;
            prev_bsdf_delta =
                has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

            // -------------------- Stopping criterion ---------------------

            dr::masked(depth, si.is_valid()) += 1;

            Float throughput_max = dr::max(unpolarized_spectrum(throughput));

            Float rr_prob    = dr::minimum(throughput_max * dr::sqr(eta), .95f);
            Mask rr_active   = depth >= m_rr_depth,
                 rr_continue = sampler->next_1d() < rr_prob;

            /* Differentiable variants of the renderer require the the russian
               roulette sampling weight to be detached to avoid bias. This is a
               no-op in non-differentiable variants. */
            throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

            active = active_next && (!rr_active || rr_continue) &&
                     dr::neq(throughput_max, 0.f);
        }

        return { /* spec  = */ dr::select(valid_ray, result, 0.f),
                 /* valid = */ valid_ray };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("BSSRDFIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

    /**
     * \brief Perform a Mueller matrix multiplication in polarized modes, and a
     * fused multiply-add otherwise.
     */
    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }
    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(BSSRDFIntegrator, MonteCarloIntegrator);
MI_EXPORT_PLUGIN(BSSRDFIntegrator, "BSSRDF integrator");
NAMESPACE_END(mitsuba)
