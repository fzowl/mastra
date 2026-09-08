import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { VoyageContextualizedEmbeddingModel, createVoyageContextualizedEmbedding } from '../contextualized-embedding';

// Mock functions
const mockContextualizedEmbed = vi.fn();
const mockConstructor = vi.fn();

// Mock the voyageai module
vi.mock('voyageai', () => {
  return {
    VoyageAIClient: class MockVoyageAIClient {
      constructor(opts: any) {
        mockConstructor(opts);
      }
      contextualizedEmbed = mockContextualizedEmbed;
    },
  };
});

/**
 * Build a contextualized_embed-shaped response: `data[docIndex].data[chunkIndex].embedding`.
 * Each document gets one embedding per chunk it contains.
 */
function mockResponse(docs: number[][][]) {
  return {
    object: 'list',
    data: docs.map((chunks, index) => ({
      object: 'list',
      index,
      data: chunks.map((embedding, chunkIndex) => ({ object: 'embedding', embedding, index: chunkIndex })),
    })),
    model: 'voyage-context-3',
    usage: { total_tokens: 1 },
  };
}

describe('VoyageContextualizedEmbeddingModel', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    vi.clearAllMocks();
    process.env = { ...originalEnv, VOYAGE_API_KEY: 'test-api-key' };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('should throw error if no API key is available', () => {
    delete process.env.VOYAGE_API_KEY;
    expect(() => new VoyageContextualizedEmbeddingModel({ model: 'voyage-context-3' })).toThrow(
      'VoyageAI API key is required',
    );
  });

  it('accepts the nested string[][] input format', async () => {
    mockContextualizedEmbed.mockResolvedValue(
      mockResponse([
        [[0.1], [0.2]],
        [[0.3]],
      ]),
    );

    const model = new VoyageContextualizedEmbeddingModel({ model: 'voyage-context-3' });
    const result = await model.doEmbed({
      values: [['doc1 chunk1', 'doc1 chunk2'], ['doc2 chunk1']],
      inputType: 'document',
    });

    expect(mockContextualizedEmbed).toHaveBeenCalledWith(
      expect.objectContaining({
        inputs: [['doc1 chunk1', 'doc1 chunk2'], ['doc2 chunk1']],
        model: 'voyage-context-3',
        inputType: 'document',
      }),
    );
    expect(result.embeddings).toEqual([[0.1], [0.2], [0.3]]);
    expect(result.chunkCounts).toEqual([2, 1]);
  });

  it('accepts the flat string[] input format and forwards it to the SDK unchanged', async () => {
    mockContextualizedEmbed.mockResolvedValue(mockResponse([[[0.1]], [[0.2]]]));

    const model = new VoyageContextualizedEmbeddingModel({ model: 'voyage-context-4' });
    const result = await model.doEmbed({
      values: ['document one', 'document two'],
      inputType: 'document',
    });

    // The flat list is passed through verbatim; the official API supports string[] inputs.
    expect(mockContextualizedEmbed).toHaveBeenCalledWith(
      expect.objectContaining({ inputs: ['document one', 'document two'] }),
    );
    expect(result.embeddings).toEqual([[0.1], [0.2]]);
    expect(result.chunkCounts).toEqual([1, 1]);
  });

  it('sorts documents by index before flattening', async () => {
    mockContextualizedEmbed.mockResolvedValue({
      data: [
        { index: 1, data: [{ embedding: [0.3], index: 0 }] },
        { index: 0, data: [{ embedding: [0.1], index: 0 }, { embedding: [0.2], index: 1 }] },
      ],
    });

    const model = new VoyageContextualizedEmbeddingModel({ model: 'voyage-context-3' });
    const result = await model.doEmbed({ values: [['a', 'b'], ['c']] });

    expect(result.embeddings).toEqual([[0.1], [0.2], [0.3]]);
    expect(result.chunkCounts).toEqual([2, 1]);
  });

  it('doEmbedGrouped returns embeddings grouped by document for flat inputs', async () => {
    mockContextualizedEmbed.mockResolvedValue(mockResponse([[[0.1]], [[0.2]]]));

    const model = new VoyageContextualizedEmbeddingModel({ model: 'voyage-context-4' });
    const result = await model.doEmbedGrouped({ values: ['document one', 'document two'] });

    expect(result.embeddingsByDocument).toEqual([[[0.1]], [[0.2]]]);
  });

  it('embedQuery wraps a single query in nested form', async () => {
    mockContextualizedEmbed.mockResolvedValue(mockResponse([[[0.5, 0.6]]]));

    const model = new VoyageContextualizedEmbeddingModel({ model: 'voyage-context-3' });
    const embedding = await model.embedQuery('what is the topic?');

    expect(mockContextualizedEmbed).toHaveBeenCalledWith(
      expect.objectContaining({ inputs: [['what is the topic?']], inputType: 'query' }),
    );
    expect(embedding).toEqual([0.5, 0.6]);
  });

  it('passes baseUrl through to the VoyageAIClient when provided', () => {
    createVoyageContextualizedEmbedding({ model: 'voyage-context-3', baseUrl: 'https://ai.mongodb.com/v1' });
    expect(mockConstructor).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: 'https://ai.mongodb.com/v1' }),
    );
  });
});
