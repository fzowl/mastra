/**
 * VoyageAI Contextualized Chunk Embedding Model
 *
 * Contextualized models produce embeddings that reflect a chunk's meaning within
 * its surrounding document, which improves retrieval over plain chunk embeddings.
 *
 * Design: each input string is embedded as its OWN independent document. The
 * batch is sent as a flat `string[]` with server-side auto-chunking enabled and
 * a chunk size of 32,000 tokens (the model's per-chunk context window), so every
 * input resolves to exactly one chunk and therefore exactly one embedding. This
 * keeps the model a drop-in text embedder (one vector per input, deterministic
 * result collection) for generic `embedMany` callers.
 *
 * Cross-input contextualization (sending the whole batch as a single document's
 * chunks) is intentionally NOT used here: generic callers pass unrelated texts,
 * so treating them as one document would leak context between them. Callers that
 * want true within-document contextualization should chunk a single document and
 * embed those chunks together in their own request.
 *
 * The Voyage API rejects auto-chunking when `input_type` is `query`, so the query
 * path sends `input_type: 'query'` without auto-chunking; any other input type is
 * treated as a document and auto-chunked.
 */

import { VoyageAIClient } from 'voyageai';
import { createTokenAwareBatches } from './batching';
import type {
  VoyageContextModel,
  VoyageContextualizedEmbeddingConfig,
  VoyageProviderOptions,
  VoyageInputType,
} from './types';

/** Maximum inputs (documents) allowed in a single contextualized request. */
const MAX_INPUTS_PER_CALL = 1000;

/** Maximum tokens allowed across a single contextualized request. */
const MAX_REQUEST_TOKENS = 120000;

/**
 * Target chunk size in tokens for auto-chunking. Set to the per-chunk context
 * window so any input within the model's limit becomes exactly one chunk.
 */
const AUTO_CHUNK_SIZE = 32000;

/**
 * Resolve embedding parameters and run a batched contextualized request.
 *
 * Returns one embedding per input, in input order.
 */
async function embedContextualized(
  client: VoyageAIClient,
  modelId: string,
  values: string[],
  config: VoyageContextualizedEmbeddingConfig,
  providerOptions?: VoyageProviderOptions,
  inputTypeOverride?: VoyageInputType,
): Promise<number[][]> {
  const inputType: VoyageInputType | undefined =
    inputTypeOverride ?? providerOptions?.voyage?.inputType ?? config.inputType ?? undefined;
  const outputDimension = providerOptions?.voyage?.outputDimension ?? config.outputDimension;
  const outputDtype = providerOptions?.voyage?.outputDtype ?? config.outputDtype;

  // The API rejects auto-chunking for queries; every other input type is treated
  // as a document and auto-chunked so each input yields exactly one chunk.
  const isQuery = inputType === 'query';
  const sdkInputType: 'query' | 'document' = isQuery ? 'query' : 'document';

  const batches = await createTokenAwareBatches(
    client,
    modelId,
    values,
    MAX_REQUEST_TOKENS,
    MAX_INPUTS_PER_CALL,
  );

  const allEmbeddings: number[][] = [];

  for (const batch of batches) {
    const response = await client.contextualizedEmbed({
      // Flat list: each string is its own independent document/query.
      inputs: batch,
      model: modelId,
      inputType: sdkInputType,
      outputDimension: outputDimension,
      outputDtype: outputDtype,
      ...(isQuery ? {} : { enableAutoChunking: true, chunkSize: AUTO_CHUNK_SIZE }),
    });

    // The SDK wraps each input as a result with its own chunk embeddings. With
    // one chunk per input, the embedding is always that document's first chunk.
    const sortedResults = [...(response.results ?? [])].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    for (const doc of sortedResults) {
      allEmbeddings.push(doc.embeddings?.[0] ?? []);
    }
  }

  return allEmbeddings;
}

/**
 * VoyageAI Contextualized Chunk Embedding Model - V2 Implementation
 *
 * Implements the EmbeddingModelV2 interface (AI SDK v5). Each input string is
 * embedded independently as its own document and returns exactly one embedding,
 * so this can be used anywhere a plain text embedder is expected.
 *
 * @example
 * ```typescript
 * const model = new VoyageContextualizedEmbeddingModelV2({ model: 'voyage-context-4' });
 *
 * // Embed documents (each string embedded independently)
 * const { embeddings } = await model.doEmbed({ values: ['First doc...', 'Second doc...'] });
 *
 * // Embed a query for retrieval
 * const query = await model.doEmbed({
 *   values: ['What was the revenue?'],
 *   providerOptions: { voyage: { inputType: 'query' } },
 * });
 * ```
 */
export class VoyageContextualizedEmbeddingModelV2 {
  readonly specificationVersion = 'v2' as const;
  readonly provider = 'voyage' as const;
  readonly modelId: string;
  readonly maxEmbeddingsPerCall = MAX_INPUTS_PER_CALL;
  readonly supportsParallelCalls = true;

  private client: VoyageAIClient;
  private config: VoyageContextualizedEmbeddingConfig;

  constructor(config: VoyageContextualizedEmbeddingConfig) {
    this.modelId = config.model;
    this.config = config;

    const apiKey = config.apiKey || process.env.VOYAGE_API_KEY;
    if (!apiKey) {
      throw new Error(
        'VoyageAI API key is required. Set VOYAGE_API_KEY environment variable or pass apiKey in config.',
      );
    }

    this.client = new VoyageAIClient({ apiKey, ...(config.baseUrl ? { baseUrl: config.baseUrl } : {}) });
  }

  /**
   * Generate one contextualized embedding per input string.
   *
   * @param args.values - Inputs to embed; each string is embedded independently
   * @param args.providerOptions - Runtime options to override config (including inputType)
   * @returns Object containing one embedding per input, in input order
   */
  async doEmbed(args: {
    values: string[];
    abortSignal?: AbortSignal;
    headers?: Record<string, string>;
    providerOptions?: VoyageProviderOptions;
  }): Promise<{ embeddings: number[][] }> {
    const embeddings = await embedContextualized(
      this.client,
      this.modelId,
      args.values,
      this.config,
      args.providerOptions,
    );
    return { embeddings };
  }
}

/**
 * VoyageAI Contextualized Chunk Embedding Model - V3 Implementation
 *
 * Implements the EmbeddingModelV3 interface (AI SDK v6). Wraps the V2
 * implementation and adds the warnings array.
 */
export class VoyageContextualizedEmbeddingModelV3 {
  readonly specificationVersion = 'v3' as const;
  readonly provider = 'voyage' as const;
  readonly modelId: string;
  readonly maxEmbeddingsPerCall = MAX_INPUTS_PER_CALL;
  readonly supportsParallelCalls = true;

  private v2Model: VoyageContextualizedEmbeddingModelV2;

  constructor(config: VoyageContextualizedEmbeddingConfig) {
    this.modelId = config.model;
    this.v2Model = new VoyageContextualizedEmbeddingModelV2(config);
  }

  async doEmbed(args: {
    values: string[];
    abortSignal?: AbortSignal;
    headers?: Record<string, string>;
    providerOptions?: VoyageProviderOptions;
  }): Promise<{ embeddings: number[][]; warnings: never[] }> {
    const result = await this.v2Model.doEmbed(args);
    return { ...result, warnings: [] };
  }
}

/**
 * Create a VoyageAI contextualized chunk embedding model (V3)
 *
 * @param config - Model configuration or model name string
 * @returns EmbeddingModelV3 compatible model
 *
 * @example
 * ```typescript
 * const model = createVoyageContextualizedEmbedding('voyage-context-4');
 * const { embeddings } = await model.doEmbed({ values: ['Doc 1...', 'Doc 2...'] });
 * ```
 */
export function createVoyageContextualizedEmbedding(
  config: VoyageContextualizedEmbeddingConfig | VoyageContextModel,
): VoyageContextualizedEmbeddingModelV3 {
  const normalizedConfig: VoyageContextualizedEmbeddingConfig = typeof config === 'string' ? { model: config } : config;
  return new VoyageContextualizedEmbeddingModelV3(normalizedConfig);
}

/**
 * Create a VoyageAI contextualized chunk embedding model (V2)
 *
 * @param config - Model configuration or model name string
 * @returns EmbeddingModelV2 compatible model
 */
export function createVoyageContextualizedEmbeddingV2(
  config: VoyageContextualizedEmbeddingConfig | VoyageContextModel,
): VoyageContextualizedEmbeddingModelV2 {
  const normalizedConfig: VoyageContextualizedEmbeddingConfig = typeof config === 'string' ? { model: config } : config;
  return new VoyageContextualizedEmbeddingModelV2(normalizedConfig);
}
