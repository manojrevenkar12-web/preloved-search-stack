import { useQuery } from "@tanstack/react-query";
import type { Bench, SearchResponse } from "./types";

/** Demo mode reads recorded searches (public/demo.json, made by scripts/demo.py). Live mode calls the service. */
const LIVE = import.meta.env.VITE_API_URL as string | undefined;
export const mode = LIVE ? "live" : "demo";

export class ApiError extends Error {}

async function get<T>(url: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiError("The search service could not be reached. Check that it is running, then try again.");
  }
  if (!res.ok) throw new ApiError(`The search service answered ${res.status}.`);
  return (await res.json()) as T;
}

async function loadBench(): Promise<Bench> {
  if (!LIVE) return { ...(await get<Omit<Bench, "source">>(`${import.meta.env.BASE_URL}demo.json`)), source: "demo" };
  const [designs, report] = await Promise.all([get<Bench["designs"]>(`${LIVE}/designs`),
    get<Bench["report"]>(`${LIVE}/report`).catch(() => null)]);
  return { source: "live", designs, report, recorded: null };
}

export function useBench() {
  return useQuery({ queryKey: ["bench"], queryFn: loadBench, staleTime: 60_000, retry: 1 });
}

export function useSearch(bench: Bench | undefined, query: string, design: string) {
  return useQuery({
    queryKey: ["search", query, design],
    enabled: !!bench && query.trim().length > 0,
    queryFn: async (): Promise<SearchResponse | null> => {
      if (bench?.recorded) return bench.recorded[query]?.[design] ?? null;
      const p = new URLSearchParams({ q: query, design });
      return get<SearchResponse>(`${LIVE}/search?${p}`);
    },
  });
}
