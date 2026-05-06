"use client";

import { ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SqlPanel } from "@/components/workspace/sql-panel";
import { cn } from "@/lib/utils";
import type { QueryPlan, TrustOverlay, ValidationResult } from "@/types/api";

function getTrustBadgeVariant(tone: TrustOverlay["badges"][number]["tone"]) {
  if (tone === "success" || tone === "warning" || tone === "danger") {
    return tone;
  }
  return "outline";
}

function trustToneDotClass(tone: TrustOverlay["badges"][number]["tone"]) {
  if (tone === "success") return "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.55)]";
  if (tone === "warning") return "bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.45)]";
  if (tone === "danger") return "bg-rose-400 shadow-[0_0_8px_rgba(251,113,133,0.45)]";
  return "bg-muted-foreground/50";
}

export function UnderstandingPanel({
  plan,
  sql,
  validation,
  trustOverlay,
  processingTrace,
}: {
  plan: QueryPlan;
  sql?: string;
  validation?: ValidationResult;
  trustOverlay?: TrustOverlay | null;
  processingTrace?: Record<string, unknown> | null;
}) {
  const [showSql, setShowSql] = useState(false);
  const [trustExpanded, setTrustExpanded] = useState(true);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Как система поняла запрос</CardTitle>
        <CardDescription>Интерпретация из семантического слоя и гибридного механизма разбора запроса.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div className="space-y-2 rounded-2xl border border-border/80 bg-black/22 p-4">
            <div className="text-sm font-medium">Метрики</div>
            <div className="flex flex-wrap gap-2">
              {plan.metrics.map((item) => (
                <Badge key={item.key}>{item.label}</Badge>
              ))}
            </div>
          </div>

          <div className="space-y-2 rounded-2xl border border-border/80 bg-black/22 p-4">
            <div className="text-sm font-medium">Измерения</div>
            <div className="flex flex-wrap gap-2">
              {plan.dimensions.length ? (
                plan.dimensions.map((item) => (
                  <Badge key={item.key} variant="outline">
                    {item.label}
                  </Badge>
                ))
              ) : (
                <div className="text-sm text-muted-foreground">Без дополнительной разбивки</div>
              )}
            </div>
          </div>

          <div className="space-y-2 rounded-2xl border border-border/80 bg-black/22 p-4">
            <div className="text-sm font-medium">Фильтры и период</div>
            <div className="text-sm text-muted-foreground">
              {plan.filters.length ? plan.filters.map((item) => `${item.label}: ${String(item.value)}`).join(", ") : "Явных фильтров нет"}
            </div>
            <div className="text-sm text-muted-foreground">
              {plan.time_range.label}: {plan.time_range.start_date} → {plan.time_range.end_date}
            </div>
          </div>

          <div className="space-y-2 rounded-2xl border border-border/80 bg-black/22 p-4">
            <div className="text-sm font-medium">Уверенность интерпретации</div>
            <Badge variant={plan.confidence >= 0.75 ? "success" : plan.confidence >= 0.6 ? "warning" : "danger"}>
              {Math.round(plan.confidence * 100)}%
            </Badge>
            {plan.needs_clarification ? (
              <div className="text-sm text-amber-300">Требуется уточнение: {plan.clarification_questions.join(" ")}</div>
            ) : null}
          </div>
        </div>

        {processingTrace ? (
          <div className="rounded-2xl border border-border/80 bg-black/22 p-4 text-sm text-muted-foreground">
            <div className="mb-2 text-sm font-medium text-foreground">Explain trace</div>
            <div>Источник интерпретации: {String((processingTrace.extraction as { effective_source?: string } | undefined)?.effective_source ?? "unknown")}</div>
            <div>Intent review: {String((processingTrace.intent_review as { adjusted?: boolean } | undefined)?.adjusted ? "корректировка применена" : "без корректировок")}</div>
            <div>SQL review: {String((processingTrace.sql_review as { allowed?: boolean } | undefined)?.allowed ?? "n/a")}</div>
          </div>
        ) : null}

        {trustOverlay ? (
          <div className="rounded-2xl border border-border/80 bg-black/22">
            <button
              type="button"
              className="flex w-full items-start justify-between gap-3 rounded-2xl p-4 text-left transition hover:bg-black/30"
              onClick={() => setTrustExpanded((v) => !v)}
            >
              <div className="min-w-0 flex-1 space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <div className="text-sm font-medium text-foreground">Доверие к ответу</div>
                  <div className="flex items-center gap-1.5" title="Индикаторы по блокам проверки (наведите на точку — подпись в бейджах ниже)">
                    {trustOverlay.badges.map((item) => (
                      <span
                        key={`dot-${item.label}-${item.value}`}
                        className={cn("inline-block h-2.5 w-2.5 shrink-0 rounded-full", trustToneDotClass(item.tone))}
                        aria-hidden
                      />
                    ))}
                  </div>
                </div>
                <div className="text-sm text-muted-foreground line-clamp-2">{trustOverlay.summary}</div>
                {trustOverlay.gemini_trust_second_pass &&
                trustOverlay.trust_score_before_gemini != null &&
                trustOverlay.trust_score_before_gemini !== trustOverlay.score_percent ? (
                  <div className="text-xs text-muted-foreground">
                    До перепроверки Gemini: {trustOverlay.trust_score_before_gemini}% → сейчас: {trustOverlay.score_percent}%
                  </div>
                ) : null}
              </div>
              <div className="flex shrink-0 flex-col items-end gap-2">
                <div className="flex flex-wrap justify-end gap-2">
                  <Badge variant={trustOverlay.confidence_level === "high" ? "success" : trustOverlay.confidence_level === "medium" ? "warning" : "danger"}>
                    {trustOverlay.score_percent}%
                  </Badge>
                  <Badge variant="outline">{trustOverlay.source_label}</Badge>
                  {trustOverlay.needs_manual_review ? <Badge variant="warning">Ручная проверка</Badge> : null}
                  {trustOverlay.gemini_trust_second_pass ? (
                    <Badge variant="outline" className="border-primary/30 text-primary">
                      Gemini
                    </Badge>
                  ) : null}
                </div>
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <span>{trustExpanded ? "Свернуть" : "Развернуть"}</span>
                  {trustExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </div>
              </div>
            </button>

            {trustExpanded ? (
              <div className="space-y-4 border-t border-border/60 px-4 pb-4 pt-3">
                <div className="flex flex-wrap gap-2">
                  {trustOverlay.badges.map((item) => (
                    <Badge key={`${item.label}-${item.value}`} variant={getTrustBadgeVariant(item.tone)}>
                      {item.label}: {item.value}
                    </Badge>
                  ))}
                </div>

                {trustOverlay.auto_corrections.length ? (
                  <div className="rounded-2xl border border-amber-500/20 bg-amber-500/10 p-3 text-sm text-amber-100">
                    <div className="mb-2 font-medium">Автозамена и восстановление смысла</div>
                    {trustOverlay.auto_corrections.map((item) => (
                      <div key={item}>- {item}</div>
                    ))}
                  </div>
                ) : null}

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl border border-border/80 bg-black/20 p-3">
                    <div className="mb-2 text-sm font-medium text-foreground">Почему системе можно верить</div>
                    <div className="space-y-1 text-sm text-muted-foreground">
                      {trustOverlay.evidence.map((item) => (
                        <div key={item}>- {item}</div>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-border/80 bg-black/20 p-3">
                    <div className="mb-2 text-sm font-medium text-foreground">Что стоит перепроверить</div>
                    <div className="space-y-1 text-sm text-muted-foreground">
                      {trustOverlay.cautions.length ? (
                        trustOverlay.cautions.map((item) => <div key={item}>- {item}</div>)
                      ) : (
                        <div>Критичных сигналов недоверия не найдено.</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        {sql && validation ? (
          <div className="space-y-4">
            <div className="grid gap-3 md:grid-cols-3">
              <div className="rounded-2xl border border-border/80 bg-black/22 p-3">
                <div className="text-xs uppercase tracking-[0.12em] text-muted-foreground">Guardrails</div>
                <div className="mt-1 text-sm">{validation.allowed ? "Разрешено к запуску" : "Заблокировано"}</div>
              </div>
              <div className="rounded-2xl border border-border/80 bg-black/22 p-3">
                <div className="text-xs uppercase tracking-[0.12em] text-muted-foreground">Оценка стоимости</div>
                <div className="mt-1 text-sm">{validation.estimated_cost !== null && validation.estimated_cost !== undefined ? validation.estimated_cost : "n/a"}</div>
              </div>
              <div className="rounded-2xl border border-border/80 bg-black/22 p-3">
                <div className="text-xs uppercase tracking-[0.12em] text-muted-foreground">Оценка строк</div>
                <div className="mt-1 text-sm">{validation.estimated_rows !== null && validation.estimated_rows !== undefined ? validation.estimated_rows : "n/a"}</div>
              </div>
            </div>
            {validation.blocked_reasons.length ? (
              <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-100">
                {validation.blocked_reasons.map((reason) => (
                  <div key={reason}>- {reason}</div>
                ))}
              </div>
            ) : null}
            <div className="flex justify-end">
              <Button variant="ghost" size="sm" onClick={() => setShowSql((current) => !current)}>
                {showSql ? <ChevronUp className="mr-2 h-4 w-4" /> : <ChevronDown className="mr-2 h-4 w-4" />}
                sql
              </Button>
            </div>
            {showSql ? (
              <div className="sql-reveal">
                <SqlPanel sql={sql} validation={validation} embedded />
              </div>
            ) : null}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
