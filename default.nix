{ pkgs ? import <nixpkgs> {}
, python ? pkgs.python38
, forTest ? true
, forDev ? true
}:
{
  cluscheckEnv = pkgs.stdenv.mkDerivation {
    name = "cluscheck-env";
    buildInputs = [
      python.pkgs.numba
      python.pkgs.numpy
    ] ++ pkgs.stdenv.lib.optionals forTest [
      python.pkgs.pytest
    ] ++ pkgs.stdenv.lib.optionals forDev [
      python.pkgs.ipython
      python.pkgs.r2pipe
      python.pkgs.graphviz
      pkgs.less
    ];
  };
}
