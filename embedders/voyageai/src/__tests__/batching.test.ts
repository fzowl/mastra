import { describe, it, expect, vi } from 'vitest';
import { createTokenAwareBatches } from '../batching';

/**
 * Build a fake VoyageAI client whose `tokenize` returns a fixed token count per
 * input. Only the fields used by `createTokenAwareBatches` are implemented.
 */
function makeClient(tokensPerText: number[]) {
  const tokenize = vi.fn((texts: string[]) =>
    Promise.resolve(texts.map((_, i) => ({ tokens: [], ids: new Array(tokensPerText[i] ?? 0) }))),
  );
  return { client: { tokenize } as any, tokenize };
}

describe('createTokenAwareBatches', () => {
  it('splits at the token boundary', async () => {
    // maxTokens = 100. Inputs of 60, 60, 30 tokens -> [60,30] would be 90 (ok),
    // but 60+60 = 120 > 100, so the second input starts a new batch.
    const { client } = makeClient([60, 60, 30]);
    const texts = ['a', 'b', 'c'];

    const batches = await createTokenAwareBatches(client, 'voyage-3.5', texts, 100, 1000);

    expect(batches).toEqual([['a'], ['b', 'c']]);
  });

  it('sends a single oversized input on its own', async () => {
    // The middle input alone exceeds maxTokens. It must still go through, alone,
    // rather than being dropped or merged.
    const { client } = makeClient([40, 500, 40]);
    const texts = ['a', 'big', 'c'];

    const batches = await createTokenAwareBatches(client, 'voyage-3.5', texts, 100, 1000);

    expect(batches).toEqual([['a'], ['big'], ['c']]);
  });

  it('respects the item-count cap even when tokens fit', async () => {
    // Plenty of token budget, but maxInputsPerBatch = 2 forces three batches.
    const { client } = makeClient([1, 1, 1, 1, 1]);
    const texts = ['a', 'b', 'c', 'd', 'e'];

    const batches = await createTokenAwareBatches(client, 'voyage-3.5', texts, 1_000_000, 2);

    expect(batches).toEqual([['a', 'b'], ['c', 'd'], ['e']]);
  });

  it('returns no batches for an empty input and does not call tokenize', async () => {
    const { client, tokenize } = makeClient([]);

    const batches = await createTokenAwareBatches(client, 'voyage-3.5', [], 100, 1000);

    expect(batches).toEqual([]);
    expect(tokenize).not.toHaveBeenCalled();
  });
});
