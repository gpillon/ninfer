@echo off
setlocal
rem Quickstart: ninfer-serve, model id qwen3.8-27b-nvfp4full-dflash2-262k (the spec: native
rem 262,144 context, no scaling). Vision + DFlash2 serving preset, host RAM prefix tier on.
rem
rem This is the B side of the DFlash2 A/B: every flag matches
rem qwen3.8-27b-nvfp4full-ram-hq-e8-2b-262k.bat except the speculative backend, so the two runs
rem differ in one variable. Two MTP flags cannot come along:
rem   --lm-head-draft  DFlash2 requires the full-vocabulary proposal head (its candidates span
rem                    the whole vocabulary), and the engine rejects the pair.
rem   --adaptive-mtp   the adaptive width controller is MTP-only; DFlash2 runs a fixed width.
rem Extra flags pass through after the preset (later duplicates override earlier ones).

set "ROOT=%~dp0"
set "SERVER=%ROOT%build-ninja\apps\ninfer-serve.exe"
rem The module is part of the nvfp4full identity's complete image, so the engine requires the
rem dflash2/* objects under this profile whatever --spec says: only the v2 artifact loads.
set "WEIGHTS=%ROOT%models\qwen3_8_27b_nvfp4full-v2.ninfer"
rem Full-precision per-request records, including the drafter's per-slot acceptance profile.
rem Override by setting NINFER_JSONL before calling; the path is anchored to this script, not to
rem whatever directory the server happens to be started from.
rem The name carries a local YYYYMMDD_HHMMSS stamp so consecutive runs keep their own records;
rem TIME pads the hour with a space below 10, so substitute a zero before slicing it.
set "ORA=%TIME: =0%"
set "DATA_ORA=%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%_%ORA:~0,2%%ORA:~3,2%%ORA:~6,2%"
if not defined NINFER_JSONL set "NINFER_JSONL=%ROOT%..\logs\traces-dflash2_%DATA_ORA%.jsonl"
set "JSONL=%NINFER_JSONL%"

if not exist "%SERVER%" (
    echo [qwen3.8-27b-nvfp4full-dflash2-262k] missing %SERVER% - run configure-ninja.ps1 then build-ninja.ps1 first
    exit /b 2
)

if not exist "%WEIGHTS%" (
    echo [qwen3.8-27b-nvfp4full-dflash2-262k] missing %WEIGHTS% - build it with tools.artifact.graft_dflash2_module
    exit /b 2
)

echo [qwen3.8-27b-nvfp4full-dflash2-262k] vision + DFlash2 k=7, native 262144 context, 0.0.0.0:8080

"%SERVER%" "%WEIGHTS%" ^
  --model-id qwen3.8-27b-nvfp4full-dflash2-262k ^
  --vision ^
  --spec dflash2 ^
  --draft-tokens 7 ^
  --host 0.0.0.0 ^
  --port 8080 ^
  --cors ^
  --preserve-thinking ^
  --max-pending-requests 50 ^
  --pending-timeout-ms 3000000 ^
  --kv-dtype hq-e8-2b ^
  --max-context 262144 ^
  --max-concurrency 6 ^
  --kv-ram-capacity 16384 ^
  --kv-capacity 550000 ^
  --request-log-jsonl "%JSONL%" ^
  %*

exit /b %ERRORLEVEL%
