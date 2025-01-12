- Setup an (empty) Julia 1.10.x installation
- Move the "Manifest.toml" and "Project.toml" into your working directory (where your notebook is located).
- Within your notebook call "using Pkg; Pkg.activate(@__DIR__);"
- Calling "Pkg.instantiate()" will now install the *exact* versions specified within the files
(- This ensures that you do not run into compatibility issues with the packages and can actually reproduce the intended results)
(- Should only be necessary once)