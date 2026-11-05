-- Run once in the Supabase SQL Editor with collection/upload processes stopped.
-- Preserves observations and the original SQLite ID for retry deduplication.
BEGIN;
LOCK TABLE public.observations IN ACCESS EXCLUSIVE MODE;

ALTER TABLE public.observations ADD COLUMN IF NOT EXISTS source_id bigint;
UPDATE public.observations SET source_id = id WHERE source_id IS NULL;
ALTER TABLE public.observations ALTER COLUMN source_id SET NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS observations_source_record_unique
    ON public.observations (source_id, record_time);

CREATE SEQUENCE IF NOT EXISTS public.observations_cloud_id_seq AS bigint;
ALTER SEQUENCE public.observations_cloud_id_seq OWNED BY public.observations.id;
ALTER TABLE public.observations ALTER COLUMN id
    SET DEFAULT nextval('public.observations_cloud_id_seq'::regclass);

-- Keep the allocator above existing IDs, including when rerunning this script.
SELECT setval(
    'public.observations_cloud_id_seq'::regclass,
    GREATEST(COALESCE((SELECT MAX(id) FROM public.observations), 0), last_value, 1),
    is_called OR EXISTS (SELECT 1 FROM public.observations)
)
FROM public.observations_cloud_id_seq;

-- The old composite primary key allowed the same id at different times.
-- Keep the earliest occurrence; allocate new cloud IDs to its duplicates.
WITH ranked AS (
    SELECT ctid AS row_tid,
           row_number() OVER (PARTITION BY id ORDER BY record_time, ctid) AS occurrence
    FROM public.observations
)
UPDATE public.observations AS observation
SET id = nextval('public.observations_cloud_id_seq'::regclass)
FROM ranked
WHERE observation.ctid = ranked.row_tid AND ranked.occurrence > 1;

CREATE UNIQUE INDEX IF NOT EXISTS observations_cloud_id_unique
    ON public.observations (id);
ALTER TABLE public.observations ALTER COLUMN objects_supported DROP NOT NULL;
GRANT USAGE, SELECT ON SEQUENCE public.observations_cloud_id_seq TO service_role;
NOTIFY pgrst, 'reload schema';
COMMIT;
