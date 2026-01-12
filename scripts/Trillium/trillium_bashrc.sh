# =============================================================================
# Trillium Bash Helpers for ts-tools
# =============================================================================
# Add this to your ~/.bashrc on Trillium:
#
# # ts-tools helpers
# source /project/rrg-aspuru/memoozd/ts-tools/scripts/Trillium/trillium_bashrc.sh
#
# =============================================================================

# Log directory for ts-tools jobs
TS_TOOLS_LOGS="/scratch/memoozd/ts-tools-scratch/logs"

# View .out file(s) for a job ID (vim)
aout () {
  local id="$1"
  [[ $id =~ ^[0-9]{6,}$ ]] || { echo "Usage: aout <jobid>  (e.g., aout 123456)"; return 2; }

  local files=("$TS_TOOLS_LOGS"/*"$id".out)
  [[ -e "${files[0]}" ]] || { echo "No .out files found for $id in $TS_TOOLS_LOGS"; return 1; }

  vim "${files[@]}"
}

# View .err file(s) for a job ID (vim)
aerr () {
  local id="$1"
  [[ $id =~ ^[0-9]{6,}$ ]] || { echo "Usage: aerr <jobid>  (e.g., aerr 123456)"; return 2; }

  local files=("$TS_TOOLS_LOGS"/*"$id".err)
  [[ -e "${files[0]}" ]] || { echo "No .err files found for $id in $TS_TOOLS_LOGS"; return 1; }

  vim "${files[@]}"
}

# Tail .out file for a running job
atout () {
  local id="$1"
  [[ $id =~ ^[0-9]{6,}$ ]] || { echo "Usage: atout <jobid>  (e.g., atout 123456)"; return 2; }

  local files=("$TS_TOOLS_LOGS"/*"$id".out)
  [[ -e "${files[0]}" ]] || { echo "No .out files found for $id in $TS_TOOLS_LOGS"; return 1; }

  tail -f "${files[@]}"
}

# Tail .err file for a running job
aterr () {
  local id="$1"
  [[ $id =~ ^[0-9]{6,}$ ]] || { echo "Usage: aterr <jobid>  (e.g., aterr 123456)"; return 2; }

  local files=("$TS_TOOLS_LOGS"/*"$id".err)
  [[ -e "${files[0]}" ]] || { echo "No .err files found for $id in $TS_TOOLS_LOGS"; return 1; }

  tail -f "${files[@]}"
}

# Cat .out file for a job ID
acout () {
  local id="$1"
  [[ $id =~ ^[0-9]{6,}$ ]] || { echo "Usage: acout <jobid>  (e.g., acout 123456)"; return 2; }

  local files=("$TS_TOOLS_LOGS"/*"$id".out)
  [[ -e "${files[0]}" ]] || { echo "No .out files found for $id in $TS_TOOLS_LOGS"; return 1; }

  cat "${files[@]}"
}

# Cat .err file for a job ID
acerr () {
  local id="$1"
  [[ $id =~ ^[0-9]{6,}$ ]] || { echo "Usage: acerr <jobid>  (e.g., acerr 123456)"; return 2; }

  local files=("$TS_TOOLS_LOGS"/*"$id".err)
  [[ -e "${files[0]}" ]] || { echo "No .err files found for $id in $TS_TOOLS_LOGS"; return 1; }

  cat "${files[@]}"
}

# List recent log files
alslogs () {
  ls -lht "$TS_TOOLS_LOGS" | head -20
}

# Activate ts-tools venv
tsvenv () {
  source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
}

# Quick cd to scratch
cdas () {
  cd /scratch/memoozd/ts-tools-scratch
}

# Quick cd to project
cdap () {
  cd /project/rrg-aspuru/memoozd/ts-tools
}
