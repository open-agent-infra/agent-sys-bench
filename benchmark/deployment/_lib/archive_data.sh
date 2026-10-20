# Sourced by benchmark/deployment/*/scripts/2.run.sh.
#
# Agents append to their trace CSVs / trajectory dirs under
# benchmark/data/<setting>, so without this a new run would be mixed with
# every previous one (and benchmark/analysis would report them together).
# Moves the previous run aside to benchmark/data/_archive/<setting>-<ts>.
# Set AGSBENCH_KEEP_DATA=1 to append to the existing data instead.
archive_previous_run() {
    local setting="$1"
    local data_root
    [ "${AGSBENCH_KEEP_DATA:-0}" = 1 ] && return 0
    data_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../data" 2>/dev/null && pwd)" || return 0
    [ -n "$(ls -A "$data_root/$setting" 2>/dev/null)" ] || return 0
    local dest="_archive/${setting}-$(date +%Y%m%d-%H%M%S)"
    echo "Archiving previous run: benchmark/data/$setting -> benchmark/data/$dest"
    # Containers write these files as root, so move them from a container too.
    docker run --rm --entrypoint sh -v "$data_root:/d" agsbench-request:v1 \
        -c "mkdir -p /d/_archive && mv '/d/$setting' '/d/$dest'"
}

# Sourced by 3.cleanup.sh: containers write benchmark/data/<setting> as root
# (some files 0600), so open it up for reading by benchmark/analysis on the host.
make_data_readable() {
    local setting="$1"
    local data_root
    data_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../data" 2>/dev/null && pwd)" || return 0
    [ -d "$data_root/$setting" ] || return 0
    docker run --rm --entrypoint chmod -v "$data_root:/d" agsbench-request:v1 \
        -R a+rX "/d/$setting" || true
}
