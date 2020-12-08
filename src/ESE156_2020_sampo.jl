using IJulia
using Conda
function sampo_jupyterlab(; port=2221, dir=joinpath(dirname(@__FILE__), ".."), detached=false)
    IJulia.inited && error("IJulia is already running")
    lab = IJulia.find_jupyter_subcommand("lab")
    jupyter = first(lab)
    if dirname(jupyter) == abspath(Conda.SCRIPTDIR) &&
       !Sys.isexecutable(exe(jupyter, "-lab")) &&
       isyes(Base.prompt("install JupyterLab via Conda, y/n? [y]"))
        Conda.add("jupyterlab")
    end
    remote_args = `--no-browser --port=$port`
    remote_lab = `$lab $remote_args`
    println(dir)
    return IJulia.launch(remote_lab, dir, detached)
end
sampo_jupyterlab()
