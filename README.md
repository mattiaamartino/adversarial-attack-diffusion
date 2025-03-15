# adversarial-attack-diffusion

# Learnable Prompt for Instruct-Pix2Pix

The repository now contains a complete and fully functional implementation of the **learnable prompt** module (just working on "cuda" or "cpu"). Whether positive or negative, prompts pass through this module, creating a **learnable context** used as the embedding input for the **Instruct-Pix2Pix** model.

## TODO

- [x] Implement the full attack class, connecting the learnable prompt in **Instruct-Pix2Pix** with **DINO-Small**.  
- [x] Test if a **learnable prompt with 0 context** and a **random prompt** injected into the model produce similar outputs compared to the model using the same random prompt (or directly compare embeddings). Check ```learnable_prompt_text.ipynb``` for the full experiment.
- [x] Build the loss that pushes one class to another and interpret the embedding learned (in the LearnablePrompt).
- [ ] Decide on the **training strategy** for the learnable prompt (one per image? shared across a class?) and implement the **training function**.  