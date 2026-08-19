import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  VoyageContextualizedEmbeddingModelV2,
  VoyageContextualizedEmbeddingModelV3,
  createVoyageContextualizedEmbedding,
  createVoyageContextualizedEmbeddingV2,
} from '../contextualized-embedding';

const mockContextualizedEmbed = vi.fn();
const mockTokenize = vi.fn();
const mockConstructor = vi.fn();

vi.mock('voyageai', () => {
  return {
    VoyageAIClient: class MockVoyageAIClient {
      constructor(opts: any) {
        mockConstructor(opts);
      }
      contextualizedEmbed = mockContextualizedEmbed;
      tokenize = mockTokenize;
    },
  };
});

/**
 * Build a contextualized result where each input becomes its own document with a
 * single chunk embedding, mirroring the SDK's wrapped `results` shape under the
 * flat auto-chunking design.
 */
function responseFor(embeddings: number[][]) {
  return {
    results: embeddings.map((embedding, index) => ({
      index,
      embeddings: [embedding],
    })),
    totalTokens: 10,
    rawResponse: {},
  };
}

describe('VoyageContextualizedEmbeddingModelV2', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    vi.clearAllMocks();
    process.env = { ...originalEnv, VOYAGE_API_KEY: 'test-api-key' };
    // One chunk-sized input per text by default.
    mockTokenize.mockImplementation((texts: string[]) =>
      Promise.resolve(texts.map(() => ({ tokens: ['a'], ids: [1] }))),
    );
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('creates a model with the expected metadata', () => {
    const model = new VoyageContextualizedEmbeddingModelV2({ model: 'voyage-context-4' });

    expect(model.specificationVersion).toBe('v2');
    expect(model.provider).toBe('voyage');
    expect(model.modelId).toBe('voyage-context-4');
    expect(model.maxEmbeddingsPerCall).toBe(1000);
  });

  it('throws when no API key is available', () => {
    delete process.env.VOYAGE_API_KEY;
    expect(() => new VoyageContextualizedEmbeddingModelV2({ model: 'voyage-context-4' })).toThrow(
      'VoyageAI API key is required',
    );
  });

  it('passes baseUrl through to the client when provided', () => {
    new VoyageContextualizedEmbeddingModelV2({
      model: 'voyage-context-4',
      apiKey: 'k',
      baseUrl: 'https://ai.mongodb.com/v1',
    });
    expect(mockConstructor).toHaveBeenCalledWith({ apiKey: 'k', baseUrl: 'https://ai.mongodb.com/v1' });
  });

  describe('document path', () => {
    it('sends a flat input list with auto-chunking enabled and one embedding per input', async () => {
      mockContextualizedEmbed.mockResolvedValue(responseFor([[0.1], [0.2]]));

      const model = new VoyageContextualizedEmbeddingModelV2({ model: 'voyage-context-4' });
      const result = await model.doEmbed({ values: ['doc one', 'doc two'] });

      expect(result.embeddings).toEqual([[0.1], [0.2]]);

      const call = mockContextualizedEmbed.mock.calls[0][0];
      // Flat list of strings, not nested chunk lists.
      expect(call.inputs).toEqual(['doc one', 'doc two']);
      expect(call.model).toBe('voyage-context-4');
      expect(call.inputType).toBe('document');
      expect(call.enableAutoChunking).toBe(true);
      expect(call.chunkSize).toBe(32000);
    });

    it('treats null/undefined input type as a document', async () => {
      mockContextualizedEmbed.mockResolvedValue(responseFor([[0.1]]));

      const model = new VoyageContextualizedEmbeddingModelV2({
        model: 'voyage-context-4',
        inputType: null,
      });
      await model.doEmbed({ values: ['doc'] });

      const call = mockContextualizedEmbed.mock.calls[0][0];
      expect(call.inputType).toBe('document');
      expect(call.enableAutoChunking).toBe(true);
    });

    it('sorts documents by index before collecting embeddings', async () => {
      mockContextualizedEmbed.mockResolvedValue({
        results: [
          { index: 2, embeddings: [[0.3]] },
          { index: 0, embeddings: [[0.1]] },
          { index: 1, embeddings: [[0.2]] },
        ],
      });

      const model = new VoyageContextualizedEmbeddingModelV2({ model: 'voyage-context-4' });
      const result = await model.doEmbed({ values: ['a', 'b', 'c'] });

      expect(result.embeddings).toEqual([[0.1], [0.2], [0.3]]);
    });
  });

  describe('query path', () => {
    it('never sends auto-chunking or chunkSize for queries', async () => {
      mockContextualizedEmbed.mockResolvedValue(responseFor([[0.9]]));

      const model = new VoyageContextualizedEmbeddingModelV2({ model: 'voyage-context-4' });
      const result = await model.doEmbed({
        values: ['what is the revenue?'],
        providerOptions: { voyage: { inputType: 'query' } },
      });

      expect(result.embeddings).toEqual([[0.9]]);

      const call = mockContextualizedEmbed.mock.calls[0][0];
      expect(call.inputs).toEqual(['what is the revenue?']);
      expect(call.inputType).toBe('query');
      expect(call.enableAutoChunking).toBeUndefined();
      expect(call.chunkSize).toBeUndefined();
    });

    it('honors query input type set on the config', async () => {
      mockContextualizedEmbed.mockResolvedValue(responseFor([[0.5]]));

      const model = new VoyageContextualizedEmbeddingModelV2({
        model: 'voyage-context-4',
        inputType: 'query',
      });
      await model.doEmbed({ values: ['query text'] });

      const call = mockContextualizedEmbed.mock.calls[0][0];
      expect(call.inputType).toBe('query');
      expect(call.enableAutoChunking).toBeUndefined();
    });
  });

  it('applies runtime output options over config', async () => {
    mockContextualizedEmbed.mockResolvedValue(responseFor([[0.1]]));

    const model = new VoyageContextualizedEmbeddingModelV2({
      model: 'voyage-context-4',
      outputDimension: 1024,
    });
    await model.doEmbed({
      values: ['doc'],
      providerOptions: { voyage: { outputDimension: 256, outputDtype: 'int8' } },
    });

    const call = mockContextualizedEmbed.mock.calls[0][0];
    expect(call.outputDimension).toBe(256);
    expect(call.outputDtype).toBe('int8');
  });

  describe('oversized inputs (server-side multi-chunk)', () => {
    it('mean-pools chunk embeddings and warns instead of dropping chunks', async () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      // One input that the server split into two chunks.
      mockContextualizedEmbed.mockResolvedValue({
        results: [{ index: 0, embeddings: [[0, 2, 4], [2, 4, 6]] }],
      });

      const model = new VoyageContextualizedEmbeddingModelV2({ model: 'voyage-context-4' });
      const result = await model.doEmbed({ values: ['a very long document...'] });

      // Still exactly one vector per input, the average of the two chunks.
      expect(result.embeddings).toEqual([[1, 3, 5]]);
      expect(warnSpy).toHaveBeenCalledTimes(1);
      expect(warnSpy.mock.calls[0]![0]).toContain('per-chunk window');

      warnSpy.mockRestore();
    });

    it('does not warn when every input yields a single chunk', async () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      mockContextualizedEmbed.mockResolvedValue(responseFor([[0.1], [0.2]]));

      const model = new VoyageContextualizedEmbeddingModelV2({ model: 'voyage-context-4' });
      await model.doEmbed({ values: ['doc one', 'doc two'] });

      expect(warnSpy).not.toHaveBeenCalled();
      warnSpy.mockRestore();
    });
  });

  it('splits into multiple requests when the token budget is exceeded', async () => {
    // Each input is 80000 tokens; the per-request budget is 120000, so two
    // inputs cannot share a request.
    mockTokenize.mockImplementation((texts: string[]) =>
      Promise.resolve(texts.map(() => ({ tokens: [], ids: new Array(80000) }))),
    );
    mockContextualizedEmbed
      .mockResolvedValueOnce(responseFor([[0.1]]))
      .mockResolvedValueOnce(responseFor([[0.2]]));

    const model = new VoyageContextualizedEmbeddingModelV2({ model: 'voyage-context-4' });
    const result = await model.doEmbed({ values: ['big one', 'big two'] });

    expect(mockContextualizedEmbed).toHaveBeenCalledTimes(2);
    expect(result.embeddings).toEqual([[0.1], [0.2]]);
  });
});

describe('VoyageContextualizedEmbeddingModelV3', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    process.env.VOYAGE_API_KEY = 'test-api-key';
    mockTokenize.mockImplementation((texts: string[]) =>
      Promise.resolve(texts.map(() => ({ tokens: ['a'], ids: [1] }))),
    );
  });

  it('returns a warnings array', async () => {
    mockContextualizedEmbed.mockResolvedValue(responseFor([[0.1]]));

    const model = new VoyageContextualizedEmbeddingModelV3({ model: 'voyage-context-4' });
    const result = await model.doEmbed({ values: ['doc'] });

    expect(result.embeddings).toEqual([[0.1]]);
    expect(result.warnings).toEqual([]);
  });
});

describe('Factory functions', () => {
  beforeEach(() => {
    process.env.VOYAGE_API_KEY = 'test-api-key';
  });

  it('createVoyageContextualizedEmbedding returns a V3 model', () => {
    const model = createVoyageContextualizedEmbedding('voyage-context-4');
    expect(model.specificationVersion).toBe('v3');
    expect(model.modelId).toBe('voyage-context-4');
  });

  it('createVoyageContextualizedEmbeddingV2 returns a V2 model', () => {
    const model = createVoyageContextualizedEmbeddingV2('voyage-context-3');
    expect(model.specificationVersion).toBe('v2');
    expect(model.modelId).toBe('voyage-context-3');
  });
});
