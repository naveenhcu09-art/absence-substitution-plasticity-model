---
title: "Absence spares, substitution diverts: a two-timescale plasticity model of expert skill under disuse and AI assistance"
---

**Raul V. Rodriguez**¹ ² \* · **Pavana Kiranmai Chepuri**¹ · **Hemachandran K**³

¹ Woxsen University, Hyderabad, Telangana, India
² Porto Business School, Porto, Portugal
³ Artificial Intelligence Research Centre (AIRC), Woxsen University, Hyderabad, Telangana, India

\* Corresponding author: Raul V. Rodriguez, Woxsen University. ORCID 0000-0001-9831-905X
Pavana Kiranmai Chepuri: ORCID 0009-0003-4174-6319

Article type: Regular Article · Running title: Absence spares, substitution diverts


## Abstract

Concern that artificial intelligence (AI) will erode human expertise assumes that working with an assistant resembles not working at all. We test this assumption with a model that combines three established mechanisms: two-timescale synaptic consolidation of a learned mapping, plasticity gated by the learner's engagement, and outcome-modulated competition between an internal pathway and an external advice input at a decision neuron. Fitted to within-player estimates of the change in rated chess performance after absences, the consolidation branch reproduces a slow within-player loss of {{b_adult}} percentage points of score per year in adults and {{b_older}} in players aged 60 and over. The age difference is captured by faster decay of the consolidated store (χ² = {{chi_shared_ts}}, 9 df) rather than by a larger unconsolidated component (χ² = {{chi_shared_b}}, 9 df), and the consolidation time is at most {{ts_adult_hi}} days. Calibrated to experiments in which learners practised with answer-supplying assistants, the substitution branch shows that a single session lowers unaided performance mainly through reduced engagement-gated plasticity, and that tutors which withhold answers cause no loss. Over months, competition at the decision neuron shifts weight to the advice input, so unaided performance falls while the stored mapping is preserved, and unaided practice restores it. We derive the fraction of unaided practice needed to keep a given share of control with the internal pathway. This reserve grows with the accuracy of the advisor. The model predicts dissociations between stored and expressed skill that can be tested with neural decoding and savings paradigms.

**Keywords:** synaptic consolidation; skill decay; three-factor plasticity; arbitration; cognitive offloading; expertise

## 1 Introduction

Skills are maintained by use. Learning changes synapses and, at a larger scale, brain structure: three months of juggling practice increased gray matter in visual-motion cortex, and the increase receded once practice stopped (Draganski et al., 2004; Boyke et al., 2008). London taxi drivers who acquired detailed knowledge of the city showed posterior hippocampal growth that trainees who failed did not (Woollett & Maguire, 2011). At the behavioural level, trained skills decline with non-use (Arthur et al., 1998; Tatel & Ackerman, 2025).

Artificial intelligence (AI) assistants raise a version of this question that use-dependence alone cannot answer. A person who works with an assistant has not stopped working. They continue to see problems, make decisions and receive outcomes, but the assistant performs some of the operations that the skill consists of. We call this *substitution*, to distinguish it from *absence*, in which the activity stops altogether. Empirical work points in different directions. Learners who practised with assistants that supplied answers performed worse once the assistant was removed, sometimes after a single session (Bastani et al., 2025; Liu et al., 2026). Endoscopists detected fewer adenomas without an AI system after months of using it (Budzyń et al., 2025). Yet a tutor that withheld answers caused no loss (Bastani et al., 2025), and senior patent lawyers who used a drafting assistant for three months improved their later unaided work (Autor et al., 2026). Macnamara et al. (2024) proposed that AI assistance may accelerate skill decay without the performer noticing.

These findings call for a mechanistic account that can say when substitution erodes capability, by how much, and what kind of loss it is. Three bodies of theory supply the ingredients. First, models of synaptic consolidation show that memories stored in synapses with several coupled timescales can be both plastic and durable (Fusi et al., 2005; Benna & Fusi, 2016), and two-timescale models capture the fast and slow components of motor adaptation (Smith et al., 2006). Second, three-factor learning rules gate synaptic change by eligibility, which depends on the neuron's own activity, and by a neuromodulatory outcome signal (Schultz et al., 1997; Frémaux & Gerstner, 2016). Plasticity therefore depends on whether a pathway is engaged, not only on whether a task is performed. Third, behaviour is arbitrated between controllers that compete for influence according to their reliability (Daw et al., 2005), and synapses converging on a neuron compete for limited resources through heterosynaptic and homeostatic mechanisms (Turrigiano, 2008; Chistiakova et al., 2014).

We combine these ingredients in a minimal model and fit or calibrate it to two kinds of data: within-player change in chess performance after absences, estimated from public rating lists, and controlled experiments on unaided performance after practice with AI. The model makes a distinction that the behavioural literature has drawn without a mechanism: between how strongly a skill is stored and how readily it is expressed (Tulving & Pearlstone, 1966; Bjork & Bjork, 1992). We show that absence and substitution act on different terms of this distinction, and we derive quantitative, testable predictions.

## 2 Model

### 2.1 Task and internal pathway

A learner classifies stimuli **x** ∈ ℝ^*N*^ (entries independent and standard normal; *N* = 200) according to a fixed rule *t* = sgn(**B**·**x**), where **B** is the task's teacher vector. The internal pathway computes a field *h* = **u**·**x**/√*N* through an expressed synaptic weight vector **u**. This teacher–student formulation is the simplest setting in which a learned mapping, its strength and its accuracy can be separated.

### 2.2 Two-timescale synaptic consolidation

Each synapse has an expressed component *u* and a consolidated component *s* (a two-variable reduction of chain models of consolidation; Benna & Fusi, 2016). Between practice trials, both evolve as

d*u*/d*t* = (*s* − *u*)/τ~u~, d*s*/d*t* = (*u* − *s*)/τ~s~ − *s*/τ~d~, (1)

where τ~u~ (5 days) is the relaxation of the expressed weight towards the consolidated one, τ~s~ the time over which the consolidated weight tracks the expressed one, and τ~d~ the slow decay of the consolidated weight. Practice changes *u* directly; *s* acquires it only through consolidation.

### 2.3 Engagement-gated plasticity

On each practice trial the expressed weights change by an error-correcting rule gated by engagement *e*:

Δ**u** = (η/√*N*) *e* [*t* − tanh(*h*)] **x**. (2)

The correct answer *t* is revealed after every trial. Engagement *e* = 1 when the learner generates a response before feedback (unaided practice, or a tutor that gives hints but not answers). When an assistant supplies the answer, the learner's own pathway is only weakly activated, and *e* = κ, with 0 ≤ κ ≤ 1. Equation 2 is the rate-based form of a three-factor rule in which presynaptic input, postsynaptic engagement and an outcome signal jointly determine plasticity (Frémaux & Gerstner, 2016).

### 2.4 Arbitration at the decision neuron

A decision neuron receives the internal field with gain *g*~I~ and, when present, an advice input *a* ∈ {−1, 1} with gain *g*~A~. The advice is correct with probability *q*. The response is sgn(*g*~I~*h* + *g*~A~*a* + ξ), with ξ ~ 𝒩(0, σ²) and σ = 0.1. After each trial, each active input receives outcome-modulated credit: its gain increases by η~g~·*e*~k~ if its own vote matched the correct answer and decreases by η~g~·*e*~k~ otherwise, where *e*~k~ is the input's activity (*e* for the internal pathway, 1 for the advice). The gains are then normalised to a fixed total, *g*~I~ + *g*~A~ = *G* = 1, with a floor of 10⁻³. Normalisation represents competition for a limited synaptic resource on the decision neuron (Turrigiano, 2008; Chistiakova et al., 2014). The rule is a minimal form of reliability-based arbitration between controllers (Daw et al., 2005).

*Unaided performance*, the quantity that empirical studies measure when the assistant is removed, is the expected accuracy on 2,000 test stimuli with the advice input absent: *p* = mean Φ(*t g*~I~*h*/σ). Because *p* depends on both the stored mapping (**u**, through *h*) and the internal gain *g*~I~, the model separates *stored* skill from *expressed* skill. We measure stored skill as *p* evaluated with *g*~I~ set to its maximum.

### 2.5 Regimes

*Practice*: trials with *e* = 1 and no advice. *Absence*: no trials; only Equation 1 runs. *Substitution*: trials with advice present and *e* = κ. *Tutor*: trials with *e* = 1 and no advice reaching the decision neuron. *Reserve*: a fraction *f* of trials, chosen at random, is unaided and the remainder is substituted.

### 2.6 Analytical results

**Absence.** For an expert whose skill has been maintained by steady practice, Equation 1 with a constant learning input gives a consolidated fraction *s*/*u* = φ = τ~d~/(τ~d~ + τ~s~) at the start of an absence. During absence the expressed skill relative to its starting value is *r*(*t*) = *M*₁₁(*t*) + *M*₁₂(*t*)φ, where *M*(*t*) is the matrix exponential of the linear system in Equation 1. Over the observed range unaided performance is approximately linear in expressed skill, so the predicted change is Δ(*t*) = −λ[1 − *r*(*t*)]. This has a small early component of relative size τ~s~/(τ~d~ + τ~s~), followed by a slow phase. When τ~d~ is much longer than the observation window, λ and τ~d~ enter only through the slow loss rate *b* (performance per year) and the early curvature, which is set by τ~s~. The absolute decay time τ~d~ is therefore not identified by absence data alone; *b* and τ~s~ are.

**Arbitration fixed point.** Let *c*~I~ = 2*p*~vote~ − 1 be the correlation of the internal pathway's vote with the correct answer and *c*~A~ = 2*q* − 1 that of the advice. With a reserve *f*, the expected credit per trial is *E c*~I~ for the internal input, with *E* = *f* + (1 − *f*)κ, and (1 − *f*)*c*~A~ for the advice. Under additive credit and multiplicative normalisation, the gains converge to the share

*g*~I~/*G* → σ* = *E c*~I~ / [*E c*~I~ + (1 − *f*) *c*~A~]. (3)

The reserve needed to keep at least a share σ~c~ with the internal pathway follows by solving σ* ≥ σ~c~ for *f*:

*f*\* = [σ~c~*c*~A~ − κ*c*~I~(1 − σ~c~)] / [(1 − κ)*c*~I~(1 − σ~c~) + σ~c~*c*~A~]. (4)

Because *f*\* increases with *c*~A~, a more accurate assistant requires a larger reserve to keep the same share of control. This holds even though the internal pathway's accuracy is unchanged.

## 3 Methods

### 3.1 Data

*Absence.* We used within-player estimates of the change in chess performance after absences of different lengths. These were computed from the International Chess Federation (FIDE) monthly standard rating lists for July 2012 to December 2019, as compiled in a public repository (Gravier & Dahiya, n.d.; CC BY 4.0), using procedures described in a companion manuscript (Rodriguez et al., manuscript in preparation). For each month in which a player played, the per-game score residual is the rating change divided by the product of the development coefficient and the number of games. It is expressed relative to continuously active players in the same month, rating band, age band and games band, in percentage points (pp) of score per game; near an expected score of 50%, 1 pp corresponds to about 7 Elo points. The estimate for an absence of a given length is a within-player difference-in-differences. It is the residual on return minus the player's mean residual in the second to sixth active months before the absence, compared with the same quantity for continuously active players. The last month before an absence is excluded because performance dips just before players stop. The estimates and their standard errors, come from a primary analysis of {{n_returns}} returns by {{n_players}} players. We fit the estimates for two age groups (20–59 and 60 and over) in six absence bins (1–2, 3–5, 6–11, 12–23, 24–47 and 48 or more months); these are given in Online Resource 1, with the player-weighted mean absence length in each bin.

*Substitution.* The calibration target is the pooled effect of practising with an AI assistant that could supply answers on later unaided performance, among learners: *d* = −0.27, 95% confidence interval [−0.45, −0.09]. It comes from a random-effects synthesis of the experiments of Bastani et al. (2025), Liu et al. (2026), Shen and Tamkin (2026) and Kazemitabaar et al. (2023), reported in the companion manuscript. The guardrailed tutor in Bastani et al. (2025) had no effect on unaided performance (*d* = −0.01).

### 3.2 Fitting the absence branch

For each age group we fitted the slow loss rate *b* and the consolidation time τ~s~ (bounded below by τ~u~ = 5 days) by weighted least squares on the six bin estimates, using the reduction in Section 2.6 evaluated at each bin's mean absence. We report profile-likelihood 95% intervals (Δχ² = 3.84), a linear-only comparison model, and two joint models: one in which the groups share τ~s~ but differ in *b*, and one in which they share *b* but differ in τ~s~. Results were unchanged when λ, which fixes τ~d~ given *b*, was varied from 50 to 1,000 (Online Resource 1).

### 3.3 Calibrating the substitution branch and simulating experts

*Learners.* {{n_learners}} simulated learners with heterogeneous learning rates (log-normal, s.d. 0.4 on the log scale) practised unaided for 120–360 days at five trials a day before an experimental session. The session consisted of 30 trials of unaided practice, tutor practice or substituted practice (*q* = 0.9). Unaided performance was then tested. Conditions used common random numbers (identical learners and stimulus sequences). With engagement fixed at κ = {{kappa}}, we chose the arbitration learning rate η~g~ so that substituted practice reproduced the pooled *d* = −0.27. We also solved for the η~g~ at the ends of the empirical interval, and we report sensitivity to κ.

*Experts.* A population of {{n_experts}} experts with 5–20 years of unaided practice (2–20 trials a day, median 8) and heterogeneous learning rates (log-normal, s.d. 0.3 on the log scale) was trained. Simulations used τ~s~ = 30 days, inside the fitted interval, and the same task and parameters as for learners. Skill was measured as the probit of unaided accuracy and expressed in units of its standard deviation across experts. We set τ~d~ so that the population's loss over one year of absence equalled the empirical adult loss in the same units: 0.73 pp a year × 6.95 Elo per pp, divided by the 304-Elo standard deviation of ratings among active adults, which gives {{target_sd}} standard deviations a year. This gave τ~d~ = {{tau_d_years}} years for the simulated population. Because this value depends on the mapping from model skill to ratings, we do not interpret it in absolute terms. Experts were then followed for 12 months under absence, tutor practice and substitution (κ = {{kappa}}, *q* = 0.9) with reserves of 0% and 20%, for arbitration learning rates of zero, the calibrated value and the upper value. Finally, we simulated two years of unaided practice after 12 months of full substitution.

*Reserve.* For *q* ∈ {0.75, 0.9, 0.99} and *f* ∈ {0, 0.1, 0.2, 0.3, 0.5}, experts were simulated for three years and the final internal gain share was compared with Equation 3.

### 3.4 Software, reproducibility and use of language models

Simulations used Python 3 with NumPy and SciPy, with fixed seeds. All code, derived data and parameter files are freely available for academic use (see Code availability). A large language model (Claude, Anthropic) assisted with drafting code and text. The authors reviewed all code, outputs and text, re-ran every analysis, and take full responsibility for the content. *[Authors: confirm or amend this statement before submission.]*

## 4 Results

### 4.1 Absence: slow decay of a consolidated store, faster with age

The two-timescale model fitted the within-player estimates in both age groups (Fig. 2a,b): χ² = {{chi_adult}} on {{df_adult}} df for adults and {{chi_older}} on {{df_older}} df for older players. The slow loss rate was {{b_adult}} pp a year (95% interval {{b_adult_ci}}) in adults and {{b_older}} pp a year ({{b_older_ci}}) in players aged 60 and over, a ratio of {{b_ratio}}. The early component was small. The consolidation time τ~s~ was at most {{ts_adult_hi}} days in adults and {{ts_older_hi}} days in older players (upper ends of the 95% profile intervals; Fig. 2c), and a linear model without an early component fitted almost equally well (χ² = {{chi_lin_adult}} and {{chi_lin_older}}). In the model this means that almost all of an expert's skill sits in the consolidated store, so absence acts on the slow decay term alone.

The two joint models discriminate between mechanisms for the age difference. When the groups shared τ~s~ and differed in *b*, the fit was good (χ² = {{chi_shared_ts}}, 9 df). When they shared *b* and could differ only in the early, unconsolidated component, it was poor (χ² = {{chi_shared_b}}, 9 df). Older players' faster loss is therefore attributed to faster decay of what has been consolidated, not to a larger share of unconsolidated skill. This is consistent with age-related changes in the maintenance of synaptic plasticity (Burke & Barnes, 2006).

### 4.2 One session of substitution: mainly reduced plasticity

With κ = {{kappa}}, the calibrated arbitration learning rate was η~g~ = {{eta_g}}. The empirical interval bounds η~g~ only from above. Rates up to {{eta_g_hi}} keep the effect above its lower end (*d* = −0.45). Even with η~g~ = 0 the effect ({{d_rep}}) is larger in magnitude than its upper end (−0.09). At the calibrated values, one session of substituted practice lowered unaided performance by *d* = {{d_sub}} relative to unaided practice (Fig. 3a). Of this, {{d_rep}} remained when the internal gain was held at its maximum. Most of the single-session loss therefore came from reduced engagement-gated plasticity, that is, from what was not learned, rather than from a shift in arbitration. The loss shrank as engagement rose (*d* = {{d_k0}} at κ = 0; {{d_k1}} at κ = 1). Tutor practice keeps engagement at 1 and keeps the answer from reaching the decision neuron. It is therefore identical to unaided practice in the model and produces no loss by construction (*d* = {{d_tutor}}). This structural prediction matches the null effect of the guardrailed tutor in Bastani et al. (2025).

### 4.3 Months of substitution: expressed skill falls, stored skill does not

Skill changes for experts are expressed in standard deviations of unaided skill across the simulated expert population, which is narrow. They are therefore large in absolute terms and should be read as comparisons between regimes, not as forecasts. Over 12 months, absence changed unaided skill by {{abs_12}} standard deviations, which is the calibration target, and continued tutor practice changed it by {{tut_12}} (Fig. 4a). Under full substitution without arbitration plasticity (η~g~ = 0), unaided skill did not fall ({{sub_none_12}}), because partial engagement and feedback kept the stored mapping in use. With arbitration plasticity, weight shifted to the advice input. The internal gain share fell to {{gI_sub_central}} after a year at the calibrated rate, and unaided skill fell by {{sub_central_12}} standard deviations, or {{sub_upper_12}} at the upper rate. A 20% reserve reduced the loss to {{sub20_central_12}}. Throughout, stored skill was preserved: at 12 months it had changed by {{rep_end}} standard deviations relative to baseline. The loss was a loss of expression, not of storage.

When unaided practice resumed (Fig. 4b), expressed skill recovered half of the loss in {{rec50}} days and 90% in {{rec90}} days, and by two years it was {{rec_end}} standard deviations above baseline. The time course was set by the arbitration learning rate, not by consolidation, and stored skill continued to rise with practice.

### 4.4 The reserve threshold depends on the advisor

After three years the simulated internal gain share was within {{res_maxdiff}} of the fixed point in Equation 3 for every combination of reserve and advisor accuracy, lying slightly above it because convergence is slow (Fig. 5a). For a fixed reserve, a more accurate assistant left the internal pathway with a smaller share: without a reserve, {{res_q75_f0}} at *q* = 0.75 and {{res_q99_f0}} at *q* = 0.99. From Equation 4 with κ = {{kappa}} and the experts' vote accuracy, keeping half of the control with the internal pathway requires a reserve of {{fstar_75}} of trials when the assistant is 75% accurate, {{fstar_90}} when it is 90% accurate and {{fstar_99}} when it is 99% accurate (Fig. 5b).

### 4.5 Learners: substitution arrests acquisition

Novice learners started at chance. After six months of practice at five trials a day, the unaided accuracy of those who practised unaided or with a tutor was {{lm_un_end}} (Fig. 3b). Those whose practice was substituted reached only {{lm_sub_end}}, and their internal gain share fell to {{lm_sub_g}}. A 20% reserve gave partial protection ({{lm_r20_end}}). This is the model's counterpart of "never-skilling" (Ke et al., 2026). For a learner, substitution prevents acquisition, and for an expert it diverts expression.

## 5 Discussion

The model separates two routes by which AI assistance can lower unaided performance and places them on different variables. Absence acts on storage, through slow decay of consolidated weights. Fitted to population-scale chess data, this decay is slow, and about {{b_ratio}} times faster after age 60 because the consolidated store decays faster, not because more skill is left unconsolidated. Substitution acts first on plasticity, because a pathway that is not engaged does not learn, and then on expression, because an advice input that is reliably right wins the competition for control of the response. The internal mapping can then remain intact but under-used. This is a mechanistic version of the distinction between storage and retrieval strength (Bjork & Bjork, 1992) and between availability and accessibility (Tulving & Pearlstone, 1966).

The account offers a reconciliation of the professional evidence. Endoscopists using a detection aid that flags polyps were rewarded for following it, and they lost unaided detection (Budzyń et al., 2025). In the model, this corresponds to arbitration plasticity acting on a well-stored skill. Senior lawyers who edited AI drafts kept generating judgements of their own, which corresponds to high engagement, and they improved (Autor et al., 2026). In the model, substitution with feedback and no arbitration shift slightly improves stored skill (Section 4.3). Whether an assistant erodes expertise therefore depends on two measurable properties of how it is used: how much the person's own pathway is engaged (κ), and whether following the assistant is reinforced at the expense of the person's own judgement (η~g~).

### 5.1 Predictions

1. *Storage–expression dissociation.* After prolonged substitution, the task mapping should remain decodable from task-relevant cortical activity (for example with multivariate pattern analysis) even when unaided performance has fallen. After an equivalent fall caused by absence, decodability should fall with performance.
2. *Savings with a characteristic time course.* Relearning after substitution should be faster than original learning. The speed should be set by how quickly reliance on the assistant is unlearned, so recovery should be accelerated by manipulations that reward the person's own responses, such as blind first judgements scored against outcomes.
3. *Engagement gating.* At equal exposure, designs that require a response before advice is shown should preserve unaided performance in proportion to the engagement they produce (Fig. 3a).
4. *Advisor accuracy.* Holding practice constant, a more accurate assistant should produce a larger fall in unaided performance and require a larger reserve (Equation 4). This is a counter-intuitive prediction that distinguishes arbitration from simple disuse.
5. *Age.* Losses after absence should scale with age through decay of consolidated skill. Losses within the first weeks of absence should be small at all ages.

### 5.2 Limitations

The task is an abstract linear classification, and the decision neuron is a single unit; the model is a minimal circuit, not an anatomical one. The absence data are behavioural and observational. The within-player estimator removes much, but not necessarily all, selection into absence, and players who never return are not observed. The absolute decay time is not identified by absence data, and the mapping from model skill to ratings rests on an assumed probit scale and on the variability of the simulated expert population. The arbitration learning rate is bounded only from above by the learner experiments, so the expert projections are given as ranges. The model does not include development, so it does not address the absence of decline in players under 20, and it does not include motivational effects such as the increase in giving up reported by Liu et al. (2026), although reduced reliance on the internal pathway offers a candidate mechanism for them. These are reasons to test the predictions above with neural measures rather than to treat the fitted values as final.

## 6 Conclusion

A minimal model built from consolidation, engagement-gated plasticity and arbitration accounts for slow, age-dependent loss of expert skill during absence. It also explains why AI assistance can lower unaided performance more quickly, and in a different way. Absence erodes what is stored; substitution first withholds the practice that would build the store and then diverts control away from it. The practical implication follows directly from Equations 3 and 4: the better the assistant, the more deliberately unaided practice must be reserved.

## Statements and Declarations

**Funding.** The authors did not receive support from any organisation for the submitted work.

**Competing interests.** The authors have no relevant financial or non-financial interests to disclose.

**Ethics approval.** This study used only publicly available, anonymised rating data and published summary statistics. No new data were collected from human participants, and no ethics approval was required.

**Data availability.** The FIDE rating lists are publicly available (Gravier & Dahiya, n.d.; CC BY 4.0). The derived estimates used for fitting (bin estimates, standard errors and mean absence lengths) are provided in Online Resource 1 and in the code repository.

**Code availability.** All model, fitting, simulation and figure code, with parameter files and fixed seeds, is freely available for academic use at [repository URL to be added on submission].

**Author contributions.** *[To be completed by the authors using the CRediT taxonomy.]*

## References

Arthur, W., Jr., Bennett, W., Jr., Stanush, P. L., & McNelly, T. L. (1998). Factors that influence skill decay and retention: A quantitative review and analysis. *Human Performance, 11*(1), 57–101. https://doi.org/10.1207/s15327043hup1101_3

Autor, D., Rodchenko, T., Martin, J., Iscenko, Z., Strand, S., Pearl, D., & Ferere, M. (2026). *Does AI assistance enhance or erode expertise? Evidence from a three-month field experiment in patent drafting* [Working paper]. Massachusetts Institute of Technology, Department of Economics. https://economics.mit.edu/sites/default/files/2026-10/Does%20AI%20Assistance%20Enhance%20or%20Erode%20Expertise%20Evidence%20from%20a%20Three-Month%20Field%20Experiment%20in%20Patent%20Drafting.pdf

Bastani, H., Bastani, O., Sungu, A., Ge, H., Kabakcı, Ö., & Mariman, R. (2025). Generative AI without guardrails can harm learning: Evidence from high school mathematics. *Proceedings of the National Academy of Sciences, 122*(26), e2422633122. https://doi.org/10.1073/pnas.2422633122

Benna, M. K., & Fusi, S. (2016). Computational principles of synaptic memory consolidation. *Nature Neuroscience, 19*(12), 1697–1706. https://doi.org/10.1038/nn.4401

Bjork, R. A., & Bjork, E. L. (1992). A new theory of disuse and an old theory of stimulus fluctuation. In A. F. Healy, S. M. Kosslyn, & R. M. Shiffrin (Eds.), *From learning processes to cognitive processes: Essays in honor of William K. Estes* (Vol. 2, pp. 35–67). Erlbaum.

Boyke, J., Driemeyer, J., Gaser, C., Büchel, C., & May, A. (2008). Training-induced brain structure changes in the elderly. *The Journal of Neuroscience, 28*(28), 7031–7035. https://doi.org/10.1523/JNEUROSCI.0742-08.2008

Budzyń, K., Romańczyk, M., Kitala, D., Kołodziej, P., Bugajski, M., Adami, H. O., Blom, J., Buszkiewicz, M., Halvorsen, N., Hassan, C., Romańczyk, T., Holme, Ø., Jarus, K., Fielding, S., Kunar, M., Pellise, M., Pilonis, N., Kamiński, M. F., Kalager, M., ... Mori, Y. (2025). Endoscopist deskilling risk after exposure to artificial intelligence in colonoscopy: A multicentre, observational study. *The Lancet Gastroenterology & Hepatology, 10*(10), 896–903. https://doi.org/10.1016/S2468-1253(25)00133-5

Burke, S. N., & Barnes, C. A. (2006). Neural plasticity in the ageing brain. *Nature Reviews Neuroscience, 7*(1), 30–40. https://doi.org/10.1038/nrn1809

Chistiakova, M., Bannon, N. M., Bazhenov, M., & Volgushev, M. (2014). Heterosynaptic plasticity: Multiple mechanisms and multiple roles. *The Neuroscientist, 20*(5), 483–498. https://doi.org/10.1177/1073858414529829

Daw, N. D., Niv, Y., & Dayan, P. (2005). Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control. *Nature Neuroscience, 8*(12), 1704–1711. https://doi.org/10.1038/nn1560

Draganski, B., Gaser, C., Busch, V., Schuierer, G., Bogdahn, U., & May, A. (2004). Changes in grey matter induced by training. *Nature, 427*(6972), 311–312. https://doi.org/10.1038/427311a

Frémaux, N., & Gerstner, W. (2016). Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules. *Frontiers in Neural Circuits, 9*, 85. https://doi.org/10.3389/fncir.2015.00085

Fusi, S., Drew, P. J., & Abbott, L. F. (2005). Cascade models of synaptically stored memories. *Neuron, 45*(4), 599–611. https://doi.org/10.1016/j.neuron.2005.02.001

Gravier, J., & Dahiya, A. (n.d.). *FIDE: Public chess profile data from the International Chess Federation, January 2001–December 2019* [Data set]. GitHub. Retrieved October 26, 2026, from https://github.com/JGravier/FIDE

Kazemitabaar, M., Chow, J., Ma, C. K. T., Ericson, B. J., Weintrop, D., & Grossman, T. (2023). Studying the effect of AI code generators on supporting novice learners in introductory programming. In *Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems* (pp. 1–23). Association for Computing Machinery. https://doi.org/10.1145/3544548.3580919

Ke, Y., Jin, L., Ong, J. C. L., Thirunavukarasu, A. J., Car, J., Cheung, C. Y., Tham, Y. C., Ting, D. S. W., Ong, M. E. H., Compton, S., Narayan, A., Keane, P. A., Wong, T. Y., Bates, D. W., Tan, P., & Liu, N. (2026). AI-induced never-skilling in medical education. *Nature Medicine, 32*(6), 1997–2006. https://doi.org/10.1038/s41591-026-04438-y

Liu, G., Christian, B., Dumbalska, T., Bakker, M. A., & Dubey, R. (2026). *AI assistance reduces persistence and hurts independent performance* (arXiv:2604.04721v6). arXiv. https://doi.org/10.48550/arXiv.2604.04721

Macnamara, B. N., Berber, I., Çavuşoğlu, M. C., Krupinski, E. A., Nallapareddy, N., Nelson, N. E., Smith, P. J., Wilson-Delfosse, A. L., & Ray, S. (2024). Does using artificial intelligence assistance accelerate skill decay and hinder skill development without performers' awareness? *Cognitive Research: Principles and Implications, 9*, 46. https://doi.org/10.1186/s41235-024-00572-8

Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science, 275*(5306), 1593–1599. https://doi.org/10.1126/science.275.5306.1593

Shen, J. H., & Tamkin, A. (2026). *How AI impacts skill formation* (arXiv:2601.20245). arXiv. https://doi.org/10.48550/arXiv.2601.20245

Smith, M. A., Ghazizadeh, A., & Shadmehr, R. (2006). Interacting adaptive processes with different timescales underlie short-term motor learning. *PLoS Biology, 4*(6), e179. https://doi.org/10.1371/journal.pbio.0040179

Tatel, C. E., & Ackerman, P. L. (2025). Procedural skill retention and decay: A meta-analytic review. *Psychological Bulletin, 151*(6), 696–736. https://doi.org/10.1037/bul0000481

Tulving, E., & Pearlstone, Z. (1966). Availability versus accessibility of information in memory for words. *Journal of Verbal Learning and Verbal Behavior, 5*(4), 381–391. https://doi.org/10.1016/S0022-5371(66)80048-8

Turrigiano, G. G. (2008). The self-tuning neuron: Synaptic scaling of excitatory synapses. *Cell, 135*(3), 422–435. https://doi.org/10.1016/j.cell.2008.10.008

Woollett, K., & Maguire, E. A. (2011). Acquiring "the Knowledge" of London's layout drives structural brain changes. *Current Biology, 21*(24), 2109–2114. https://doi.org/10.1016/j.cub.2011.11.018

## Figure captions

**Fig. 1** Model. The internal pathway maps stimuli to a field through expressed synaptic weights **u**, which are coupled to consolidated weights **s** that decay slowly. Practice changes **u** through an engagement-gated delta rule. A decision neuron combines the internal field and an external advice input with gains that compete for a fixed resource and are updated by outcome-modulated credit.

**Fig. 2** Absence. **a, b** Within-player change in chess performance after absences, by age group (points: estimates with 95% confidence intervals; lines: fitted two-timescale model). **c** Profile Δχ² for the consolidation time τ~s~; the dashed line marks the 95% threshold.

**Fig. 3** Learners. **a** Effect of one 30-trial session of practice with answer-supplying advice on unaided performance, as a function of engagement κ: total effect, and the effect with the internal gain held at its maximum (stored skill only). The band and dashed line show the pooled empirical estimate. **b** Unaided accuracy over six months of practice under unaided, tutor, substituted and substituted-with-reserve practice.

**Fig. 4** Experts. **a** Change in unaided skill over 12 months under absence, tutor practice and substitution with reserves of 0% and 20%. Lines use the calibrated arbitration learning rate; bands span zero to the upper bound. **b** Recovery during unaided practice after 12 months of full substitution: expressed skill and stored skill.

**Fig. 5** Reserve. **a** Internal gain share after three years of substitution as a function of the reserve, for three advisor accuracies (lines: Equation 3; circles: simulation). **b** Minimum reserve needed to keep a given share of control with the internal pathway, as a function of advisor accuracy (Equation 4).
