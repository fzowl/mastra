/**
 * VoyageAI Contextualized Chunk Embedding Model
 *
 * Contextualized models produce embeddings that reflect a chunk's meaning within
 * its surrounding document, which improves retrieval over plain chunk embeddings.
 *
 * Design: each input string is embedded as its OWN independent document. The
 * batch is sent as a flat `string[]` with server-side auto-chunking enabled and
 * a chunk size of 32,000 tokens (the model's per-chunk context window), so every
 * input within that window resolves to exactly one chunk and therefore exactly
 * one embedding. This keeps the model a drop-in text embedder (one vector per
 * input, deterministic result collection) for generic `embedMany` callers.
 *
 * An input longer than the per-chunk window is split into multiple chunks by the
 * server. To preserve the one-vector-per-input contract without dropping the
 * extra chunks, those chunk embeddings are averaged into a single vector and a
 * warning is emitted (callers should pre-chunk or shorten such inputs).
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
import { CONTEXTUALIZED_MODEL_INFO } from './types';

/** Maximum inputs (documents) allowed in a single contextualized request. */
const MAX_INPUTS_PER_CALL = 1000;

/** Fallback per-request token budget when a model is missing from the info table. */
const DEFAULT_MAX_REQUEST_TOKENS = 120000;

/** Fallback auto-chunk size (per-chunk context window) for unknown models. */
const DEFAULT_AUTO_CHUNK_SIZE = 32000;

/**
 * Average a document's chunk embeddings into a single vector. Used only for the
 * rare input that exceeds the per-chunk window and is split server-side, so the
 * whole input is represented instead of silently dropping every chunk but the
 * first. Cosine similarity is scale-invariant, so the un-normalized mean is a
 * usable document vector.
 */
function meanPool(chunks: number[][]): number[] {
  const first = chunks[0] ?? [];
  const dims = first.length;
  const pooled = new Array<number>(dims).fill(0);
  for (const chunk of chunks) {
    for (let d = 0; d < dims; d++) {
      pooled[d]! += chunk[d] ?? 0;
    }
  }
  for (let d = 0; d < dims; d++) {
    pooled[d]! /= chunks.length;
  }
  return pooled;
}

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

  // Per-model limits, sourced from the info table like the text path (rather than
  // hardcoded): `maxRequestTokens` is the whole-request budget used for batching,
  // `maxInputTokens` is the per-chunk window used as the auto-chunk size.
  const info = CONTEXTUALIZED_MODEL_INFO[modelId as VoyageContextModel];
  const maxRequestTokens = info?.maxRequestTokens ?? DEFAULT_MAX_REQUEST_TOKENS;
  const autoChunkSize = info?.maxInputTokens ?? DEFAULT_AUTO_CHUNK_SIZE;

  // The API rejects auto-chunking for queries; every other input type is treated
  // as a document and auto-chunked so each input yields exactly one chunk.
  const isQuery = inputType === 'query';
  const sdkInputType: 'query' | 'document' = isQuery ? 'query' : 'document';

  const batches = await createTokenAwareBatches(client, modelId, values, maxRequestTokens, MAX_INPUTS_PER_CALL);

  const allEmbeddings: number[][] = [];

  for (const batch of batches) {
    const response = await client.contextualizedEmbed({
      // Flat list: each string is its own independent document/query.
      inputs: batch,
      model: modelId,
      inputType: sdkInputType,
      outputDimension: outputDimension,
      outputDtype: outputDtype,
      ...(isQuery ? {} : { enableAutoChunking: true, chunkSize: autoChunkSize }),
    });

    // The SDK wraps each input as a result with its own chunk embeddings. Inputs
    // within the per-chunk window yield exactly one chunk. An input larger than
    // the window is split into multiple chunks server-side; rather than silently
    // keeping only the first chunk (dropping the rest of the input), warn and
    // mean-pool the chunks so the whole input is represented in one vector.
    const sortedResults = [...(response.results ?? [])].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    for (const doc of sortedResults) {
      const chunks = doc.embeddings ?? [];
      if (chunks.length > 1) {
        console.warn(
          `[@mastra/voyageai] A contextualized input for "${modelId}" exceeded the ${autoChunkSize}-token ` +
            `per-chunk window and was split into ${chunks.length} chunks. Its chunk embeddings were averaged ` +
            `into one vector. Pre-chunk or shorten inputs to keep one embedding per input.`,
        );
        allEmbeddings.push(meanPool(chunks));
      } else {
        allEmbeddings.push(chunks[0] ?? []);
      }
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
