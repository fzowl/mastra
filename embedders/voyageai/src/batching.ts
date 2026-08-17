/**
 * Token-aware batching for VoyageAI embedding requests.
 *
 * VoyageAI caps each request by two independent limits: a maximum number of
 * inputs and a maximum number of tokens across the whole request. Batching only
 * by item count can still overflow the token budget for long inputs, so this
 * builds batches that respect both limits.
 */

import type { VoyageAIClient } from 'voyageai';

/**
 * Split texts into batches that respect both the per-request token budget and
 * the per-request input-count cap.
 *
 * Uses the SDK's `tokenize()` for accurate per-input token counts. A single
 * input whose token count already exceeds `maxTokens` is sent on its own rather
 * than being dropped, matching the behavior of the reference LangChain client.
 *
 * @param client - VoyageAI client used to tokenize inputs
 * @param model - Model id whose tokenizer should be used
 * @param texts - Inputs to batch
 * @param maxTokens - Maximum tokens allowed across a single request
 * @param maxInputsPerBatch - Maximum inputs allowed in a single request
 * @returns Array of batches, each an array of inputs
 */
export async function createTokenAwareBatches(
  client: VoyageAIClient,
  model: string,
  texts: string[],
  maxTokens: number,
  maxInputsPerBatch: number,
): Promise<string[][]> {
  if (texts.length === 0) return [];

  const tokenResults = await client.tokenize(texts, model);

  const batches: string[][] = [];
  let currentBatch: string[] = [];
  let currentTokens = 0;

  for (let i = 0; i < texts.length; i++) {
    const tokenCount = tokenResults[i]?.ids.length ?? 0;

    // Flush the current batch before adding this input if it would exceed either
    // limit. The `currentBatch.length > 0` guard ensures an oversized single
    // input still goes through alone instead of being stranded.
    if (
      currentBatch.length > 0 &&
      (currentTokens + tokenCount > maxTokens || currentBatch.length >= maxInputsPerBatch)
    ) {
      batches.push(currentBatch);
      currentBatch = [];
      currentTokens = 0;
    }

    currentBatch.push(texts[i]!);
    currentTokens += tokenCount;
  }

  if (currentBatch.length > 0) {
    batches.push(currentBatch);
  }

  return batches;
}
