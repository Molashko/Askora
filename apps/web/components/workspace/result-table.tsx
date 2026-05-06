"use client";

import { ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { formatNumber } from "@/lib/utils";
import { buildColumnLabels, formatAnalyticsValue } from "@/lib/presentation";
import type { QueryPlan } from "@/types/api";

export function ResultTable({
  columns,
  rows,
  plan,
}: {
  columns: string[];
  rows: Record<string, unknown>[];
  plan?: QueryPlan;
}) {
  const [expanded, setExpanded] = useState(false);
  const labels = buildColumnLabels(plan);

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-3 space-y-0 pb-3">
        <CardTitle className="text-base md:text-lg">Табличный результат</CardTitle>
        <Button variant="ghost" size="sm" className="shrink-0 gap-1.5 text-muted-foreground" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "Свернуть" : "Развернуть"}
          {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </Button>
      </CardHeader>
      {expanded ? (
        <CardContent className="pt-0">
          {rows.length ? (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    {columns.map((column) => (
                      <TableHead key={column}>{labels[column] ?? column}</TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((row, index) => (
                    <TableRow key={index}>
                      {columns.map((column) => (
                        <TableCell key={column}>
                          {formatNumber((typeof row[column] === "string" ? formatAnalyticsValue(row[column]) : row[column]) as number | string | null)}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <div className="rounded-2xl border border-border/80 bg-black/24 p-6 text-sm text-muted-foreground">
              Нет данных для отображения. Либо запрос был заблокирован, либо вернул пустой результат.
            </div>
          )}
        </CardContent>
      ) : (
        <CardContent className="pb-4 pt-0 text-xs text-muted-foreground">Таблица скрыта. Нажмите «Развернуть», чтобы снова показать строки.</CardContent>
      )}
    </Card>
  );
}
