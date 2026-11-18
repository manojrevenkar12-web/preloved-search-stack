export type Listing = {
  item_id: number; title: string; price: number; market: string; listed_day: number;
  brand: string | null; category: string | null; model: string | null; material: string | null;
  colour: string | null; size: string | null; condition: number | null;
};

export type SearchResponse = {
  query: string; design: string; understood: Record<string, string> | null; corrected: [string, string][];
  relaxed: string[]; candidates: number; ms: number; items: Listing[];
};

export type DesignInfo = { name: string; understand: boolean; index: string; hard: string[]; vectors: string | null;
  fusion: string; ranker: string | null; depth: number };

export type Online = { purchases_per_search: number; purchase_lift: number; gmv_lift: number;
  relevant_in_top10: number; zero_results: number; p95_ms: number; searches: number };

export type StudyRow = { kendall_tau: number; p: number; picks: string; sign_agreement: number };

export type Report = {
  world: string; control: string; proposed: string; designs: string[]; variants: string[];
  offline: Record<string, Record<string, number>>; online: Record<string, Online>;
  study: Record<string, StudyRow | string>;
  ab_summary: { replications: number; rollout_lift: number; mean_lift: number; spread: number; covered: number;
    significant: number; max_interference: number; cuped_variance_reduction: number };
  aa: { plain: number; cuped: number; splits: number };
  efficiency: { ratio: number; interleaving: { searches_needed: number }; ab_cuped: { searches_needed: number } };
  gate: Record<string, boolean>;
};

export type Bench = {
  source: "demo" | "live";
  designs: DesignInfo[];
  recorded: Record<string, Record<string, SearchResponse>> | null;  // demo only: query -> design -> response
  report: Report | null;
};
