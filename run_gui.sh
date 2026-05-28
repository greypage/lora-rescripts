#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export HF_HOME="${HF_HOME:-huggingface}"
export PYTHONUTF8=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

resolve_runtime_dir() {
    local preferred_name="$1"
    shift
    local env_root="$script_dir/env"
    local dir_name

    if [[ -d "$env_root" ]]; then
        for dir_name in "$@"; do
            if [[ -d "$env_root/$dir_name" ]]; then
                printf '%s\n' "$env_root/$dir_name"
                return 0
            fi
        done
    fi

    for dir_name in "$@"; do
        if [[ -d "$script_dir/$dir_name" ]]; then
            printf '%s\n' "$script_dir/$dir_name"
            return 0
        fi
    done

    if [[ -d "$env_root" ]]; then
        printf '%s\n' "$env_root/$preferred_name"
    else
        printf '%s\n' "$script_dir/$preferred_name"
    fi
}

portable_runtime_dir="$(resolve_runtime_dir "python" "python")"
portable_python="$portable_runtime_dir/bin/python"
venv_runtime_dir="$(resolve_runtime_dir "venv" "venv" ".venv" ".local-runtime/venv")"
venv_python="$venv_runtime_dir/bin/python"
portable_marker="$portable_runtime_dir/.deps_installed"
venv_marker="$venv_runtime_dir/.deps_installed"
sage_runtime_dir="$(resolve_runtime_dir "python-sageattention" "python-sageattention" "python_sageattention")"
sage_runtime_dir_name="$(basename "$sage_runtime_dir")"
sage_python="$sage_runtime_dir/bin/python"
sage_marker="$sage_runtime_dir/.deps_installed"
tageditor_portable_runtime_dir="$(resolve_runtime_dir "python_tageditor" "python_tageditor")"
tageditor_portable_python="$tageditor_portable_runtime_dir/bin/python"
tageditor_venv_runtime_dir="$(resolve_runtime_dir "venv-tageditor" "venv-tageditor")"
tageditor_venv_python="$tageditor_venv_runtime_dir/bin/python"
allow_external_python="${MIKAZUKI_ALLOW_SYSTEM_PYTHON:-0}"
preferred_runtime="${MIKAZUKI_PREFERRED_RUNTIME:-}"
prefer_sageattention_runtime=0
if [[ "$preferred_runtime" == "sageattention" ]]; then
    prefer_sageattention_runtime=1
fi
main_modules=(accelerate torch fastapi toml transformers diffusers lion_pytorch dadaptation schedulefree prodigyopt prodigyplus pytorch_optimizer)

python_exe=""
deps_marker=""
tageditor_python=""
tageditor_marker=""
disable_tageditor=0
runtime_name="standard"

find_system_python() {
    if [[ -n "${PYTHON:-}" ]]; then
        if [[ -x "${PYTHON}" ]]; then
            printf '%s\n' "${PYTHON}"
            return 0
        fi
        if command -v "${PYTHON}" >/dev/null 2>&1; then
            command -v "${PYTHON}"
            return 0
        fi
    fi

    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi

    echo "No usable system Python was found. Install python3 or set \$PYTHON first." >&2
    exit 1
}

test_pip_ready() {
    local python_bin="$1"
    "$python_bin" -m pip --version >/dev/null 2>&1
}

test_sageattention_runtime_ready() {
    local python_bin="$1"

    "$python_bin" -c "import importlib.metadata as md; import torch
try:
    import triton  # noqa: F401
    from sageattention import sageattn, sageattn_varlen
    ok = callable(sageattn) and callable(sageattn_varlen) and torch.cuda.is_available()
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)" >/dev/null 2>&1
}

set_dedicated_runtime_caches() {
    local current_runtime_name="$1"
    local python_bin="$2"

    if [[ "$current_runtime_name" != "sageattention" ]]; then
        return 0
    fi

    local runtime_root
    runtime_root="$(cd "$(dirname "$python_bin")/.." >/dev/null 2>&1 && pwd)"
    local cache_root="$runtime_root/.cache"
    local triton_cache_dir="${TRITON_CACHE_DIR:-$cache_root/triton}"
    local torchinductor_cache_dir="${TORCHINDUCTOR_CACHE_DIR:-$cache_root/torchinductor}"

    mkdir -p "$cache_root" "$triton_cache_dir" "$torchinductor_cache_dir"

    export TRITON_CACHE_DIR="$triton_cache_dir"
    export TORCHINDUCTOR_CACHE_DIR="$torchinductor_cache_dir"
    export TRITON_HOME="${TRITON_HOME:-$cache_root}"

    echo "Persistent compile cache enabled for $current_runtime_name runtime:"
    echo "- TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
    echo "- TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
}

test_modules_ready() {
    local python_bin="$1"
    shift

    if [[ "$#" -eq 0 ]]; then
        return 0
    fi

    "$python_bin" -c "import importlib, sys; failed=[]
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except Exception:
        failed.append(name)
raise SystemExit(1 if failed else 0)" "$@" >/dev/null 2>&1
}

test_package_constraints() {
    local python_bin="$1"
    shift

    if [[ "$#" -eq 0 ]]; then
        return 0
    fi

    "$python_bin" -c "import sys, importlib.metadata as md; from packaging.specifiers import SpecifierSet; from packaging.version import Version; ok=True
for item in sys.argv[1:]:
    name, spec = item.split(chr(31), 1)
    try:
        version = md.version(name)
    except md.PackageNotFoundError:
        ok = False
        continue
    if spec and Version(version) not in SpecifierSet(spec):
        ok = False
raise SystemExit(0 if ok else 1)" "$@" >/dev/null 2>&1
}

explain_modules_ready() {
    local python_bin="$1"
    shift

    if [[ "$#" -eq 0 ]]; then
        return 0
    fi

    "$python_bin" -c "import importlib, sys, traceback
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
        print(f'[OK] import {name}')
    except Exception as exc:
        print(f'[FAIL] import {name}: {exc}')
        traceback.print_exc()" "$@" 2>&1
}

explain_package_constraints() {
    local python_bin="$1"
    shift

    if [[ "$#" -eq 0 ]]; then
        return 0
    fi

    "$python_bin" -c "import sys, importlib.metadata as md
from packaging.specifiers import SpecifierSet
from packaging.version import Version
for item in sys.argv[1:]:
    name, spec = item.split(chr(31), 1)
    try:
        version = md.version(name)
    except md.PackageNotFoundError:
        print(f'[MISSING] {name} {spec}'.strip())
        continue
    if spec and Version(version) not in SpecifierSet(spec):
        print(f'[MISMATCH] {name}=={version} does not satisfy {spec}')
    else:
        print(f'[OK] {name}=={version}')" "$@" 2>&1
}

report_tageditor_dependency_state() {
    local python_bin="$1"
    local marker_path="$2"
    shift 2
    local modules=("$@")
    local constraints=(
        $'gradio\x1f==4.28.3'
        $'gradio-client\x1f==0.16.0'
        $'fastapi\x1f<0.113'
        $'starlette\x1f<0.39'
        $'pydantic\x1f<2.11'
        $'huggingface-hub\x1f<1'
    )

    echo "Tag editor dependency check target: $python_bin" >&2
    if [[ -n "$marker_path" ]]; then
        if [[ -f "$marker_path" ]]; then
            echo "Tag editor marker file: OK ($marker_path)" >&2
        else
            echo "Tag editor marker file is missing: $marker_path" >&2
        fi
    else
        echo "Tag editor marker file: not used for this runtime selection" >&2
    fi

    echo "Tag editor import check details:" >&2
    explain_modules_ready "$python_bin" "${modules[@]}" >&2 || true

    echo "Tag editor package constraint details:" >&2
    explain_package_constraints "$python_bin" "${constraints[@]}" >&2 || true
}

get_python_minor_version() {
    local python_bin="$1"
    "$python_bin" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null
}

runtime_uses_sageattention() {
    local current_runtime_name="$1"
    [[ "$current_runtime_name" == "sageattention" ]]
}

selected_runtime_dependencies_ready() {
    local current_runtime_name="$1"
    local python_bin="$2"
    local marker_path="$3"

    if ! test_modules_ready "$python_bin" "${main_modules[@]}"; then
        return 1
    fi

    if [[ -n "$marker_path" && ! -f "$marker_path" ]]; then
        return 1
    fi

    if runtime_uses_sageattention "$current_runtime_name" && ! test_sageattention_runtime_ready "$python_bin"; then
        return 1
    fi

    return 0
}

install_selected_runtime_dependencies() {
    local current_runtime_name="$1"

    if [[ "$current_runtime_name" == "sageattention" || "$prefer_sageattention_runtime" -eq 1 ]]; then
        echo "SageAttention experimental dependencies are not installed yet. Running install_sageattention.sh..."
        bash "$script_dir/install_sageattention.sh"
        return $?
    fi

    echo "Dependencies are not installed yet. Running install.bash..."
    bash "$script_dir/install.bash"
}

get_selected_runtime_install_failure_message() {
    local current_runtime_name="$1"

    if runtime_uses_sageattention "$current_runtime_name"; then
        echo "SageAttention dependency installation failed."
        return 0
    fi

    echo "Dependency installation failed."
}

select_main_python() {
    if [[ "$prefer_sageattention_runtime" -eq 1 ]]; then
        if [[ -x "$sage_python" ]]; then
            echo "Using SageAttention experimental Python..."
            if ! test_pip_ready "$sage_python"; then
                echo "SageAttention runtime is incomplete: pip is not available. Repair or recreate ./$sage_runtime_dir_name first." >&2
                exit 1
            fi
            python_exe="$sage_python"
            deps_marker="$sage_marker"
            runtime_name="sageattention"
            return 0
        fi

        echo "SageAttention startup was requested, but the dedicated runtime is missing. Running install_sageattention.sh..." >&2
        bash "$script_dir/install_sageattention.sh"
        if [[ -x "$sage_python" ]] && test_pip_ready "$sage_python"; then
            echo "Using SageAttention experimental Python..."
            python_exe="$sage_python"
            deps_marker="$sage_marker"
            runtime_name="sageattention"
            return 0
        fi

        echo "SageAttention runtime bootstrap failed. Expected: $sage_python" >&2
        exit 1
    fi

    if [[ -x "$portable_python" ]]; then
        echo "Using portable Python..."
        if ! test_pip_ready "$portable_python"; then
            echo "Portable Python is incomplete: pip is not available. Repair or replace the bundled python folder first." >&2
            exit 1
        fi
        python_exe="$portable_python"
        deps_marker="$portable_marker"
        runtime_name="portable"
        return 0
    fi

    if [[ -x "$venv_python" ]]; then
        echo "Using virtual environment..."
        python_exe="$venv_python"
        deps_marker="$venv_marker"
        runtime_name="venv"
        return 0
    fi

    if [[ "$allow_external_python" == "1" ]]; then
        echo "No project-local Python found. MIKAZUKI_ALLOW_SYSTEM_PYTHON=1 is set, bootstrapping a project-local venv via install.bash..."
        bash "$script_dir/install.bash"
        if [[ -x "$portable_python" ]]; then
            python_exe="$portable_python"
            deps_marker="$portable_marker"
            runtime_name="portable"
            return 0
        fi
        if [[ -x "$venv_python" ]]; then
            python_exe="$venv_python"
            deps_marker="$venv_marker"
            runtime_name="venv"
            return 0
        fi
        echo "install.bash finished, but no project-local Python environment was created." >&2
        exit 1
    fi

    cat >&2 <<EOF
No project-local Python environment was found.

This build is locked to project-local Python by default to avoid leaking installs into the host machine.

Expected one of:
- $portable_python
- $venv_python

Recommended fix:
1. Bundle a ready-to-run portable Python in ./env/python or the legacy ./python
2. Or set MIKAZUKI_ALLOW_SYSTEM_PYTHON=1 and rerun to bootstrap a project-local ./env/venv or legacy ./venv for development
EOF
    exit 1
}

select_tageditor_python() {
    tageditor_python=""
    tageditor_marker=""

    if [[ -x "$tageditor_portable_python" ]]; then
        tageditor_python="$tageditor_portable_python"
        tageditor_marker="$tageditor_portable_runtime_dir/.tageditor_installed"
        return 0
    fi

    if [[ -x "$tageditor_venv_python" ]]; then
        tageditor_python="$tageditor_venv_python"
        tageditor_marker="$tageditor_venv_runtime_dir/.tageditor_installed"
        return 0
    fi

    local fallback_main_python=""
    if [[ "$runtime_name" == "sageattention" ]]; then
        if [[ -x "$portable_python" ]]; then
            fallback_main_python="$portable_python"
            tageditor_marker="$portable_runtime_dir/.tageditor_installed"
        elif [[ -x "$venv_python" ]]; then
            fallback_main_python="$venv_python"
            tageditor_marker="$venv_runtime_dir/.tageditor_installed"
        fi
    else
        fallback_main_python="$python_exe"
        if [[ "$python_exe" == "$portable_python" ]]; then
            tageditor_marker="$portable_runtime_dir/.tageditor_installed"
        elif [[ "$python_exe" == "$venv_python" ]]; then
            tageditor_marker="$venv_runtime_dir/.tageditor_installed"
        fi
    fi

    local main_python_version
    if [[ -n "$fallback_main_python" ]]; then
        main_python_version="$(get_python_minor_version "$fallback_main_python" || true)"
    else
        main_python_version=""
    fi

    if [[ -n "$main_python_version" && "$main_python_version" != "3.13" ]]; then
        tageditor_python="$fallback_main_python"
    else
        tageditor_marker=""
    fi
}

for arg in "$@"; do
    if [[ "$arg" == "--disable-tageditor" ]]; then
        disable_tageditor=1
        break
    fi
done

select_main_python
set_dedicated_runtime_caches "$runtime_name" "$python_exe"

main_install_needed=0

if ! selected_runtime_dependencies_ready "$runtime_name" "$python_exe" "$deps_marker"; then
    main_install_needed=1
fi

if [[ "$main_install_needed" -eq 1 ]]; then
    install_selected_runtime_dependencies "$runtime_name"
    select_main_python
    set_dedicated_runtime_caches "$runtime_name" "$python_exe"
    if ! selected_runtime_dependencies_ready "$runtime_name" "$python_exe" "$deps_marker"; then
        get_selected_runtime_install_failure_message "$runtime_name" >&2
        exit 1
    fi
fi

if [[ "$disable_tageditor" -eq 0 ]]; then
    select_tageditor_python
    if [[ -n "$tageditor_python" ]]; then
        tageditor_modules=(gradio transformers timm print_color)
        tageditor_constraints=(
            $'gradio\x1f==4.28.3'
            $'gradio-client\x1f==0.16.0'
            $'fastapi\x1f<0.113'
            $'starlette\x1f<0.39'
            $'pydantic\x1f<2.11'
            $'huggingface-hub\x1f<1'
        )
        tageditor_install_needed=0
        tageditor_modules_ready=1
        tageditor_constraints_ready=1
        tageditor_marker_ready=1

        if ! test_modules_ready "$tageditor_python" "${tageditor_modules[@]}"; then
            tageditor_install_needed=1
            tageditor_modules_ready=0
        fi

        if [[ -n "$tageditor_marker" && ! -f "$tageditor_marker" ]]; then
            tageditor_marker_ready=0
        fi

        if [[ "$tageditor_install_needed" -eq 0 && "$tageditor_marker_ready" -eq 0 ]]; then
            tageditor_install_needed=1
        fi

        if [[ "$tageditor_install_needed" -eq 0 ]] && ! test_package_constraints "$tageditor_python" "${tageditor_constraints[@]}"; then
            tageditor_install_needed=1
            tageditor_constraints_ready=0
        fi

        if [[ "$tageditor_install_needed" -eq 1 ]]; then
            echo "Tag editor dependencies are missing or incompatible for Python: $tageditor_python" >&2
            report_tageditor_dependency_state "$tageditor_python" "$tageditor_marker" "${tageditor_modules[@]}"
            if ! test_pip_ready "$tageditor_python"; then
                echo "Tag editor Python is incomplete: pip is not available." >&2
                exit 1
            fi

            echo "Tag editor dependencies are missing or incompatible. Running install_tageditor.sh..."
            bash "$script_dir/install_tageditor.sh"
            select_tageditor_python
            if [[ -z "$tageditor_python" ]]; then
                echo "Tag editor dependency installation failed: no usable tag editor Python was selected after install." >&2
                exit 1
            fi
            if ! test_modules_ready "$tageditor_python" "${tageditor_modules[@]}" || ! test_package_constraints "$tageditor_python" "${tageditor_constraints[@]}"; then
                echo "Tag editor dependency installation failed for Python: $tageditor_python" >&2
                report_tageditor_dependency_state "$tageditor_python" "$tageditor_marker" "${tageditor_modules[@]}"
                echo "Tag editor dependency installation failed." >&2
                exit 1
            fi
            if [[ -n "$tageditor_marker" && ! -f "$tageditor_marker" ]]; then
                echo "Tag editor dependency installation failed: marker file was not created." >&2
                report_tageditor_dependency_state "$tageditor_python" "$tageditor_marker" "${tageditor_modules[@]}"
                echo "Tag editor dependency installation failed." >&2
                exit 1
            fi
        fi
    fi
fi

cd "$script_dir"
export MIKAZUKI_SKIP_REQUIREMENTS_VALIDATION=1
exec "$python_exe" gui.py "$@"
