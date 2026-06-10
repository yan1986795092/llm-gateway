# LLM Gateway

Production-grade LLM gateway with auto-routing, rate limiting, circuit breaker, and Redis cache.

## Features

- **OpenAI-compatible API** — drop-in replacement for OpenAI API
- **Auto-routing** — 3D weighted scoring (cost, latency, quality)
- **Multi-provider** — OpenAI, Anthropic, Gemini, Ollama
- **Rate limiting** — sliding window algorithm
- **Circuit breaker** — protects against cascading failures
- **Redis cache** — request deduplication
- **Observability** — structured logging, tracing-ready

## Quick Start

### Prerequisites

- Python 3.11+
- Redis
- API keys for your LLM providers

### Local Development

```bash
cp .env.example .env
# Edit .env with your API keys

pip install -e ".[dev]"
uvicorn gateway.main:app --reload --port 9010
```

### Docker

```bash
cp .env.example .env
# Edit .env with your API keys

docker-compose up --build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /v1/chat/completions | Chat completions (OpenAI compatible) |
| GET | /v1/models | List available models |
| GET | /health | Health check |
| GET | /metrics | Prometheus metrics |

## Configuration

See `.env.example` for all available environment variables.

## Architecture

```
Request → Rate Limiter → Cache → Circuit Breaker → Router → Provider Adapter → Response
```

## License

MIT
