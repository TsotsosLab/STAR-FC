#ifndef FACTORIZE_H
#define FACTORIZE_H

#include <iostream>

// Code adapted from gsl/fft/factorize.c
void factorize (const int n, int *n_factors, int factors[], int * implemented_factors);
bool is_optimal(int n, int * implemented_factors);
int find_closest_factor(int n, int * implemented_factor);

#endif
