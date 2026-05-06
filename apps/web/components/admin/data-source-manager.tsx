"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Database, Trash2, UploadCloud } from "lucide-react";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/api";
import type { DataSourceSummary } from "@/types/api";

const manualSourceSchema = z.object({
  key: z.string().min(2),
  name: z.string().min(2),
  description: z.string().optional(),
  dialect: z.string().min(2),
  connection_url: z.string().min(5),
  schema_name: z.string().optional(),
  allowed_roles: z.string().optional(),
});

const csvSchema = z.object({
  file: z.custom<FileList>((value) => value instanceof FileList && value.length > 0, "Выберите CSV-файл"),
  display_name: z.string().optional(),
  source_key: z.string().optional(),
  delimiter: z.string().min(1).max(8),
  activate: z.boolean().default(true),
  use_llm: z.boolean().default(true),
});

function sourceCapabilities(source: DataSourceSummary): Record<string, unknown> {
  return source.capabilities_json ?? {};
}

function sourceKind(source: DataSourceSummary): string {
  const caps = sourceCapabilities(source);
  return typeof caps.kind === "string" ? caps.kind : "external";
}

function sourceTable(source: DataSourceSummary): string {
  const caps = sourceCapabilities(source);
  return typeof caps.table_name === "string" ? caps.table_name : source.schema_name || "не задано";
}

function sourceRows(source: DataSourceSummary): number | null {
  const value = sourceCapabilities(source).row_count;
  return typeof value === "number" ? value : null;
}

function sourceColumns(source: DataSourceSummary): Array<Record<string, unknown>> {
  const value = sourceCapabilities(source).columns;
  return Array.isArray(value) ? (value as Array<Record<string, unknown>>) : [];
}

export function DataSourceManager() {
  const queryClient = useQueryClient();
  const { data } = useQuery({
    queryKey: ["admin", "data-sources"],
    queryFn: api.dataSources,
  });

  const sources = data ?? [];
  const activeSource = sources.find((source) => source.is_default) ?? sources[0] ?? null;
  const uploadedSources = sources.filter((source) => sourceKind(source) === "uploaded_csv");

  const manualForm = useForm<z.infer<typeof manualSourceSchema>>({
    resolver: zodResolver(manualSourceSchema),
    defaultValues: {
      key: "default",
      name: "Основной PostgreSQL",
      description: "",
      dialect: "postgres",
      connection_url: "postgresql+psycopg://postgres:postgres@db:5432/analytics_hub",
      schema_name: "analytics",
      allowed_roles: "admin, analyst, business_user",
    },
  });

  const csvForm = useForm<z.infer<typeof csvSchema>>({
    resolver: zodResolver(csvSchema),
    defaultValues: {
      file: undefined,
      display_name: "",
      source_key: "",
      delimiter: "auto",
      activate: true,
      use_llm: true,
    },
  });

  const manualMutation = useMutation({
    mutationFn: (values: z.infer<typeof manualSourceSchema>) =>
      api.createDataSource({
        key: values.key.trim(),
        name: values.name.trim(),
        description: values.description?.trim() || "",
        dialect: values.dialect.trim(),
        connection_url: values.connection_url.trim(),
        schema_name: values.schema_name?.trim() || "",
        is_active: true,
        is_default: false,
        allowed_roles_json: values.allowed_roles
          ? values.allowed_roles
              .split(",")
              .map((item) => item.trim())
              .filter(Boolean)
          : [],
        capabilities_json: {
          scheduler: true,
          guardrails: true,
        },
      }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["admin", "data-sources"] });
      manualForm.reset();
    },
  });

  const csvMutation = useMutation({
    mutationFn: async (values: z.infer<typeof csvSchema>) => {
      const file = values.file[0];
      const payload = new FormData();
      payload.append("file", file);
      payload.append("delimiter", values.delimiter);
      payload.append("auto_mode", "true");
      payload.append("apply", "true");
      payload.append("activate", String(values.activate));
      payload.append("use_llm", String(values.use_llm));
      if ((values.display_name || "").trim()) {
        payload.append("display_name", values.display_name!.trim());
      }
      if ((values.source_key || "").trim()) {
        payload.append("source_key", values.source_key!.trim());
      }
      return api.autoConfigFromCsv(payload);
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["admin", "data-sources"] });
    },
  });

  const activateMutation = useMutation({
    mutationFn: api.activateDataSource,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["admin", "data-sources"] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: api.deleteDataSource,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["admin", "data-sources"] });
      await queryClient.invalidateQueries({ queryKey: ["dataset-context"] });
    },
  });

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5 text-primary" />
            Активный датасет
          </CardTitle>
          <CardDescription>Этот датасет сейчас использует workspace, semantic layer, SQL-builder и guardrails.</CardDescription>
        </CardHeader>
        <CardContent>
          {activeSource ? (
            <div className="rounded-2xl border border-primary/25 bg-primary/8 p-4">
              <div className="flex flex-wrap items-center gap-2">
                <div className="text-lg font-semibold">{activeSource.name}</div>
                <Badge variant="success">Используется сейчас</Badge>
                <Badge variant="outline">{sourceKind(activeSource) === "uploaded_csv" ? "CSV" : activeSource.dialect}</Badge>
              </div>
              <div className="mt-2 text-sm text-muted-foreground">{activeSource.description || "Описание не заполнено"}</div>
              <div className="mt-3 grid gap-3 text-sm md:grid-cols-3">
                <div className="rounded-xl border border-border/80 bg-black/24 p-3">
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">Таблица</div>
                  <div className="mt-1 break-all">{sourceTable(activeSource)}</div>
                </div>
                <div className="rounded-xl border border-border/80 bg-black/24 p-3">
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">Строки</div>
                  <div className="mt-1">{sourceRows(activeSource)?.toLocaleString("ru-RU") ?? "неизвестно"}</div>
                </div>
                <div className="rounded-xl border border-border/80 bg-black/24 p-3">
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">Ключ</div>
                  <div className="mt-1 break-all">{activeSource.key}</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-100">Активный датасет пока не найден.</div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <UploadCloud className="h-5 w-5 text-primary" />
            Загрузить новый CSV
          </CardTitle>
          <CardDescription>Файл будет импортирован в отдельную таблицу, а система создаст semantic layer под его колонки.</CardDescription>
        </CardHeader>
        <CardContent>
          <form className="grid gap-4 lg:grid-cols-[1fr_220px]" onSubmit={csvForm.handleSubmit((values) => csvMutation.mutate(values))}>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2 md:col-span-2">
                <Label>CSV-файл</Label>
                <Input
                  type="file"
                  accept=".csv,text/csv"
                  onChange={(event) => csvForm.setValue("file", event.target.files as FileList, { shouldValidate: true })}
                />
              </div>
              <div className="space-y-2">
                <Label>Название в админке</Label>
                <Input {...csvForm.register("display_name")} placeholder="Продажи за апрель" />
              </div>
              <div className="space-y-2">
                <Label>Ключ датасета</Label>
                <Input {...csvForm.register("source_key")} placeholder="sales_april" />
              </div>
              <div className="space-y-2">
                <Label>Delimiter</Label>
                <Input {...csvForm.register("delimiter")} placeholder="auto | , | ; | tab" />
              </div>
              <div className="space-y-3 pt-7 text-sm text-muted-foreground">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={Boolean(csvForm.watch("activate"))}
                    onChange={(event) => csvForm.setValue("activate", event.target.checked)}
                  />
                  Сразу сделать активным
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={Boolean(csvForm.watch("use_llm"))}
                    onChange={(event) => csvForm.setValue("use_llm", event.target.checked)}
                  />
                  Улучшить подписи через Gemini
                </label>
              </div>
            </div>

            <div className="flex flex-col justify-end gap-3">
              {csvMutation.error ? (
                <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 p-3 text-sm text-rose-200">{csvMutation.error.message}</div>
              ) : null}
              <Button disabled={csvMutation.isPending} size="lg">
                {csvMutation.isPending ? "Импортируем…" : "Загрузить и адаптировать"}
              </Button>
            </div>
          </form>

          {csvMutation.data ? (
            <div className="mt-4 rounded-2xl border border-border/80 bg-black/28 p-4 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant={csvMutation.data.applied ? "success" : "outline"}>{csvMutation.data.applied ? "Импортировано" : "Предпросмотр"}</Badge>
                <span className="text-muted-foreground">Delimiter: {csvMutation.data.used_delimiter}</span>
                <span className="text-muted-foreground">Dataset: {csvMutation.data.catalog_preview.base_dataset}</span>
              </div>
              <div className="mt-2 text-muted-foreground">{csvMutation.data.auto_resolution.validation_message}</div>
              <div className="mt-3 grid gap-2 md:grid-cols-3">
                <div>Метрик: {csvMutation.data.catalog_preview.metrics_count}</div>
                <div>Измерений: {csvMutation.data.catalog_preview.dimensions_count}</div>
                <div>Фильтров: {csvMutation.data.catalog_preview.filters_count}</div>
              </div>
              <div className="mt-3 max-h-36 overflow-auto rounded-xl border border-border/70 p-3 text-xs text-muted-foreground">
                {csvMutation.data.catalog_preview.columns.slice(0, 18).map((column) => (
                  <div key={column.name}>
                    {column.name} — {column.inferred_type}, filled {Math.round(column.non_null_ratio * 100)}%
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Доступные датасеты</CardTitle>
          <CardDescription>Можно переключаться между загруженными CSV и базовым demo-источником.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {sources.map((source) => {
            const columns = sourceColumns(source);
            const isUploaded = sourceKind(source) === "uploaded_csv";
            return (
              <div key={source.id} className="rounded-2xl border border-border/80 bg-black/24 p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <div className="font-medium">{source.name}</div>
                  {source.is_default ? <Badge variant="success">Активный</Badge> : <Badge variant="outline">{isUploaded ? "CSV" : source.dialect}</Badge>}
                  {!source.is_active ? <Badge variant="secondary">Выключен</Badge> : null}
                </div>
                <div className="mt-2 text-sm text-muted-foreground">{source.description || "Описание не заполнено"}</div>
                <div className="mt-3 rounded-xl border border-border/80 bg-black/30 px-3 py-2 text-xs text-muted-foreground">
                  {sourceTable(source)}
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                  <span>Ключ: {source.key}</span>
                  <span>Строки: {sourceRows(source)?.toLocaleString("ru-RU") ?? "неизвестно"}</span>
                  <span>Колонки: {columns.length || "неизвестно"}</span>
                </div>
                {columns.length ? (
                  <div className="mt-3 flex max-h-20 flex-wrap gap-2 overflow-hidden">
                    {columns.slice(0, 16).map((column) => (
                      <Badge key={String(column.name)} variant="secondary">
                        {String(column.name)} · {String(column.inferred_type)}
                      </Badge>
                    ))}
                  </div>
                ) : null}
                <div className="mt-4 flex flex-wrap gap-2">
                  {(isUploaded || source.key === "default") && !source.is_default ? (
                    <Button size="sm" variant="outline" disabled={activateMutation.isPending} onClick={() => activateMutation.mutate(source.id)}>
                      <CheckCircle2 className="mr-2 h-4 w-4" />
                      Сделать активным
                    </Button>
                  ) : null}
                  {isUploaded ? (
                    <Button
                      size="sm"
                      variant="danger"
                      disabled={deleteMutation.isPending}
                      onClick={() => {
                        if (window.confirm(`Удалить датасет «${source.name}» и его таблицу?`)) {
                          deleteMutation.mutate(source.id);
                        }
                      }}
                    >
                      <Trash2 className="mr-2 h-4 w-4" />
                      Удалить
                    </Button>
                  ) : null}
                </div>
              </div>
            );
          })}
          {deleteMutation.error ? (
            <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 p-3 text-sm text-rose-200">
              {deleteMutation.error.message}
            </div>
          ) : null}
          {!uploadedSources.length ? (
            <div className="rounded-2xl border border-border/80 bg-black/20 p-4 text-sm text-muted-foreground">
              Загруженных CSV пока нет. После импорта они появятся здесь и смогут стать активным датасетом.
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Внешний источник вручную</CardTitle>
          <CardDescription>Для подключения существующей БД.</CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={manualForm.handleSubmit((values) => manualMutation.mutate(values))}>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label>Ключ источника</Label>
                <Input {...manualForm.register("key")} placeholder="taxi_prod" />
              </div>
              <div className="space-y-2">
                <Label>Диалект</Label>
                <Input {...manualForm.register("dialect")} placeholder="postgres | mysql | clickhouse" />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Название</Label>
              <Input {...manualForm.register("name")} placeholder="Боевая витрина заказов" />
            </div>
            <div className="space-y-2">
              <Label>Описание</Label>
              <Textarea rows={2} {...manualForm.register("description")} placeholder="Описание источника" />
            </div>
            <div className="space-y-2">
              <Label>Connection URL</Label>
              <Textarea rows={3} {...manualForm.register("connection_url")} placeholder="postgresql+psycopg://..." />
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label>Schema</Label>
                <Input {...manualForm.register("schema_name")} placeholder="analytics" />
              </div>
              <div className="space-y-2">
                <Label>Роли с доступом</Label>
                <Input {...manualForm.register("allowed_roles")} placeholder="admin, analyst, business_user" />
              </div>
            </div>
            {manualMutation.error ? <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 p-3 text-sm text-rose-200">{manualMutation.error.message}</div> : null}
            <Button disabled={manualMutation.isPending}>{manualMutation.isPending ? "Сохраняем…" : "Добавить источник"}</Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
