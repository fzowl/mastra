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
 * Rough token estimate used when the SDK's local tokenizer is unavailable.
 * ~4 characters per token is the common heuristic; batching only needs an
 * estimate to decide where to split, not exact counts.
 */
function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

/**
 * Count tokens per input, preferring the SDK's `tokenize()` for accuracy and
 * falling back to a character-based estimate when it is unavailable.
 *
 * The SDK's `tokenize()` runs a local tokenizer that depends on the optional
 * `@huggingface/transformers` package; if it (or any other dependency) is not
 * installed, we still need batching to work rather than throw on every request.
 */
async function countTokens(client: VoyageAIClient, model: string, texts: string[]): Promise<number[]> {
  try {
    const tokenResults = await client.tokenize(texts, model);
    return texts.map((text, i) => tokenResults[i]?.ids.length ?? estimateTokens(text));
  } catch {
    return texts.map(estimateTokens);
  }
}

/**
 * Split texts into batches that respect both the per-request token budget and
 * the per-request input-count cap.
 *
 * Uses the SDK's `tokenize()` for accurate per-input token counts, falling back
 * to a character-based estimate when the local tokenizer is unavailable. A single
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

  const tokenCounts = await countTokens(client, model, texts);

  const batches: string[][] = [];
  let currentBatch: string[] = [];
  let currentTokens = 0;

  for (let i = 0; i < texts.length; i++) {
    const tokenCount = tokenCounts[i] ?? 0;

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
