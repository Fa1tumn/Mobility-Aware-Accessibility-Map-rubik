-- Optional: enable uploading the full Area + Material metadata.
-- Without these columns, the uploader keeps metadata locally and sends legacy fields.
BEGIN;
ALTER TABLE public.observations
    ADD COLUMN IF NOT EXISTS area_accept_ratio double precision,
    ADD COLUMN IF NOT EXISTS material_accept_ratio double precision,
    ADD COLUMN IF NOT EXISTS model_version text,
    ADD COLUMN IF NOT EXISTS policy_version text;
NOTIFY pgrst, 'reload schema';
COMMIT;
