{
  description = "Virtual Environment for Reinforcement Learning";
  inputs.systems.url = "github:nix-systems/default";

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    systems,
  }:
    flake-utils.lib.eachSystem (import systems)
    (system: let
      pkgs = import nixpkgs {
        inherit system;
      };
      lib = pkgs.lib;
    in {
      packages = flake-utils.lib.flattenTree {
        inherit (pkgs) hello;
      };

      devShells.default = let
        pythonPackages = pkgs.python311Packages;
      in pkgs.mkShell {
        venvDir = "./.venv";
        NIX_LD_LIBRARY_PATH = lib.makeLibraryPath (with pkgs; [
          stdenv.cc.cc.lib
          stdenv.cc.cc
          libcxx
          openssl
        ]);
        NIX_LD = pkgs.runCommand "ld.so" {} ''
          ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
        '';
        buildInputs = with pkgs; [
          pythonPackages.python
          pythonPackages.venvShellHook
          pythonPackages.ipython
          pythonPackages.numpy
          pythonPackages.pandas
          pythonPackages.matplotlib
          pythonPackages.torch
          pythonPackages.gymnasium
          pythonPackages.ipdb
          pythonPackages.pytest
          pythonPackages.pytest-cov
        ];
      };
    });
}
