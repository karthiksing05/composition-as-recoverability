# Diffusion tests (plan and predictions, written 2026-11-20 before any result)

Why: composition in text-to-image diffusion is the most cited case of machine composition after
language models. The paper's two tests carry over directly. The system's own procedure is what the
diffusion model does: generating an image from a caption, or choosing between captions by its own
denoising error (a diffusion classifier). The state is the denoiser's internal features. We score
both on parts held out from the readout's fit, with the same splits, part codes and kappa as the
image-encoder tests (e8_vision / e8_grid / e17_encoder_behavior).

Models (wave 1): SD 1.5 (U-Net, CLIP-L text), SDXL base (U-Net, CLIP-L + OpenCLIP-bigG).
Wave 2 if wave 1 runs clean: a DiT with a T5 text encoder (PixArt-alpha or SD 3.5 medium), FLUX.1-schnell.

## D1. Generation of relations (behavior, weak test)
Captions "a x on/under/above/below/left of/right of a y" over held-out nouns from the paper's
305-noun list (imageable subset), each with its converse paraphrase. 4 seeds per caption.
Score with GroundingDINO (cached): both objects present; relation correct from box centers;
converse agreement; word-order bias = share where the first-mentioned noun takes the first slot
(top or left) regardless of meaning.
Prediction: presence of both objects high (>0.6), relation accuracy near chance (0.5 among
images with both objects) for "on/under" and "left/right", and a word-order bias above 0.5.

## D2. The denoiser's state during generation (representation)
(a) Text side: readout of the figure from the conditioning the U-Net receives (pooled text tokens),
canonical -> converse transfer, as for text encoders. Prediction: transfer near 0 (word order),
as the paper already found for CLIP-L.
(b) Image side: for the images of D1, the U-Net's spatial features at a fixed timestep, pooled per
half of the image (top/bottom or left/right), read by a linear probe fitted on fitting nouns for
"which noun is here". Prediction: kappa well above 0 for the generated layout; the state follows
what was drawn, so errors in D1 are errors of reading the caption, not of keeping parts apart.

## D3. Two photographs on a canvas (the image-encoder test, run on diffusion models)
Same Tiny-ImageNet composites (beside / above / overlap), same 120/30/50 class split.
State: U-Net features of the noised VAE latent at timesteps t in {100, 300, 500}, empty caption,
mid block and up blocks, pooled per quadrant (grid) and over the whole map (mean).
Own procedure: the diffusion classifier (Li et al. 2023; Clark & Jaini 2023): pick between the
caption with the right places and the caption with places swapped (and, as a control, a caption
with one wrong object) by the lower denoising error averaged over timesteps and noise draws.
Prediction: grid-state kappa 0.7-0.95 (like CLIP/SigLIP quadrant states); mean-pooled kappa lower;
diffusion classifier well above chance on the wrong-object control but near chance (<0.6) on
swapped places. That puts diffusion models below the diagonal of the main figure, like CLIP.

## D4. Score composition (composable diffusion, Liu et al. 2022)
Compose two concepts by summing guidance terms, eps(x|"a x") + eps(x|"a y") - eps(x).
The sum is symmetric in x and y, so the paper's theory calls it a blend in roles: no procedure can
tell which concept fills which role, while both concepts can still be present.
Tests: (a) presence of both objects (behavior); (b) role readout from the composed image's
features: kappa for "which noun is in role 1" should be ~0 by construction, while a set readout
(both nouns present, order-free) is high; (c) composing "a x on the left" + "a y on the right"
to see whether spatial words restore roles.
Prediction: (a) high, (b) role kappa ~0 and set recovery high, (c) partial.

## D5. Trained from scratch (wave 2; diffusion-composition-how ShapeWorld U-Nets)
Parts held out from the system itself are only possible here. Read each object slot's color and
shape from the U-Net state on captions with new color-shape pairs, and compare with the model's
own generation accuracy on the same captions, across caption coverage. Prediction: low-coverage
models omit objects in generation while their states still carry each named object (behavior
lags state); to be specified after D1-D4.

## D6. Origins (H4) for diffusion
VAE latent (input to the U-Net), U-Net with random weights, trained U-Net, on D3 composites.
Prediction: the VAE latent is spatial, so its grid kappa is already high; the trained U-Net adds
little on top of the latent for places, as the glyph networks inherit from their input.

Every number: held-out classes or nouns only, 3 seeds of the readout split, validation parts choose
layer, timestep and ridge penalty.
