import { openAIResponsesApi } from "../api/openai-responses.lazy.js";
import { envApiKeyAuth } from "../auth/helpers.js";
import { createProvider } from "../models.js";
import { OPENAI_MODELS } from "./openai.models.js";
export function openaiProvider() {
    const customBaseUrl = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
    const patchedModels = Object.values(OPENAI_MODELS).map(m => ({...m, baseUrl: customBaseUrl}));
    return createProvider({
        id: "openai",
        name: "OpenAI",
        baseUrl: customBaseUrl,
        auth: { apiKey: envApiKeyAuth("OpenAI API key", ["OPENAI_API_KEY"]) },
        models: patchedModels,
        api: openAIResponsesApi(),
    });
}
