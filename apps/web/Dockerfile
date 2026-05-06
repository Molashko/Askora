FROM node:20-bookworm-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-bookworm-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules /app/node_modules
COPY . .
ARG NEXT_PUBLIC_API_URL=/api/proxy
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build

FROM node:20-bookworm-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
# Для app/api/proxy (Route Handler): куда проксировать на рантайме в Docker-сети
ENV INTERNAL_API_BASE_URL=http://api:8000
COPY --from=builder /app/.next /app/.next
COPY --from=builder /app/public /app/public
COPY --from=builder /app/node_modules /app/node_modules
COPY --from=builder /app/package.json /app/package.json
EXPOSE 3000
CMD ["npm", "run", "start"]
