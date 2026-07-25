/**
 * In-memory emulation of the backend's job registry + SSE progress surface, so
 * static-mode long-running work (the geo fine-tune worker) flows through the
 * SAME `{ready:false, job_id}` → subscribeProgress/pollJob → done-payload
 * protocol the views already implement against the live backend.
 */

import type { JobSnapshot } from "../dataClient";
import { notFoundError, toApiError } from "./errors";

export type ProgressFn = (progress: number, message: string) => void;

export interface JobHandlers {
  onProgress?: ProgressFn;
  onDone?: (data?: Record<string, unknown>) => void;
  onError?: (type: string, message: string) => void;
}

interface LocalJob {
  id: string;
  cacheKey: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  message: string;
  error: { type: string; message: string } | null;
  result: Record<string, unknown> | undefined;
  version: number;
  listeners: Set<JobHandlers>;
  settled: Promise<Record<string, unknown>>;
}

export class LocalJobRegistry {
  private readonly jobs = new Map<string, LocalJob>();
  private counter = 0;

  /**
   * Register + start a job. `run` receives a progress reporter and resolves to
   * the done payload (delivered verbatim to subscribeProgress onDone — e.g.
   * `{weights_token, loss_before, loss_after}` for fine-tunes, exactly like the
   * backend's SSE done event).
   */
  create(cacheKey: string, run: (report: ProgressFn) => Promise<Record<string, unknown>>): string {
    const id = `static-job-${++this.counter}`;
    const job: LocalJob = {
      id,
      cacheKey,
      status: "running",
      progress: 0,
      message: "starting",
      error: null,
      result: undefined,
      version: 0,
      listeners: new Set(),
      settled: Promise.resolve({}),
    };
    // Defer the runner one macrotask so callers can attach subscribers first
    // (and so a synchronous fallback runner cannot complete before the caller
    // even receives the job id).
    job.settled = new Promise<Record<string, unknown>>((resolve, reject) => {
      setTimeout(() => {
        run((p, m) => this.report(job, p, m)).then(
          (result) => {
            job.status = "done";
            job.progress = 1;
            job.message = "done";
            job.result = result;
            job.version++;
            for (const h of [...job.listeners]) {
              h.onProgress?.(1, job.message);
              h.onDone?.(result);
            }
            job.listeners.clear();
            resolve(result);
          },
          (e: unknown) => {
            const err = toApiError(e);
            job.status = "error";
            job.error = { type: err.type, message: err.message };
            job.version++;
            for (const h of [...job.listeners]) h.onError?.(err.type, err.message);
            job.listeners.clear();
            reject(err);
          },
        );
      }, 0);
    });
    // A job nobody awaits must not surface an unhandled rejection.
    job.settled.catch(() => {});
    this.jobs.set(id, job);
    return id;
  }

  snapshot(jobId: string): JobSnapshot {
    const job = this.jobs.get(jobId);
    if (!job) throw notFoundError(`job ${jobId} not found`);
    return {
      job_id: job.id,
      cache_key: job.cacheKey,
      status: job.status,
      progress: job.progress,
      message: job.message,
      error: job.error,
      version: job.version,
    };
  }

  /** Await completion, reporting progress along the way (pollJob semantics). */
  async wait(jobId: string, onProgress?: ProgressFn): Promise<Record<string, unknown>> {
    const job = this.jobs.get(jobId);
    if (!job) throw notFoundError(`job ${jobId} not found`);
    let unsub: (() => void) | undefined;
    if (onProgress) unsub = this.subscribe(jobId, { onProgress });
    try {
      const result = await job.settled;
      onProgress?.(1, job.message);
      return result;
    } finally {
      unsub?.();
    }
  }

  /**
   * subscribeProgress semantics: replays terminal state asynchronously if the
   * job already finished; returns an unsubscribe function.
   */
  subscribe(jobId: string, handlers: JobHandlers): () => void {
    const job = this.jobs.get(jobId);
    if (!job) {
      queueMicrotask(() => handlers.onError?.("NotFoundError", `job ${jobId} not found`));
      return () => {};
    }
    if (job.status === "done" || job.status === "error") {
      queueMicrotask(() => {
        if (job.status === "done") {
          handlers.onProgress?.(1, job.message);
          handlers.onDone?.(job.result);
        } else {
          handlers.onError?.(job.error?.type ?? "ComputeError", job.error?.message ?? "job failed");
        }
      });
      return () => {};
    }
    job.listeners.add(handlers);
    handlers.onProgress?.(job.progress, job.message);
    return () => job.listeners.delete(handlers);
  }

  private report(job: LocalJob, progress: number, message: string): void {
    if (job.status !== "running") return;
    job.progress = Math.max(0, Math.min(1, progress));
    job.message = message;
    job.version++;
    for (const h of [...job.listeners]) h.onProgress?.(job.progress, job.message);
  }
}
