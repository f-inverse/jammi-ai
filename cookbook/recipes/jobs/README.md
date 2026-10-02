# Jobs

Submit long-running work, watch it, cancel it, and pick it up from another
process.

**When to use this pattern.** Training and compute verbs run as rows of the
engine's durable job queue. You need to see a job's state, stop one, run
submission and execution in different processes, or clean up finished jobs.

## Run it

```bash
python cookbook/recipes/jobs/example.py
```

Seconds on CPU.
