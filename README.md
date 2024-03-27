# Gamma source effective thick calculator

## Model

### Efficiency dependency from material

The full-energy peaks efficiency of the gamma-rays detection (just efficiency) of volume sources can be written as:

$$
eff(E) = eff_{p0}(E) \frac{1 - e^{-\mu(E) \rho t}}{\mu(E) \rho t}
$$

where:
- *E* -- energy of gamma-ray
- $eff_{p0}$ -- the efficiency of the point source at the same distance,
- $\mu$ -- mass attenuation coefficient of the volume source material, cm^2/g,
- $\rho$ -- density of the volume source material, g/cm^3,
- *t* -- effective thickness, it is close to the real volume source thickness, it minimize the difference between real and calculated efficiencies. The precise value of *t* is unknown and it needs to be found.

### Convert efficiency from one material to another:

Knowing the thickness (t) you can recalulate efficiency from one material to another:

$$
eff(E) = eff_{0}(E) \frac{(1 - e^{-\mu(E) \rho t})\mu_0(E) \rho_0}{(1 - e^{-\mu_0(E) \rho_0 t}) \mu(E) \rho}
$$

where:
- $eff$ -- target efficiency of the source with material with $\mu, \rho$
- $eff_0$ -- source efficiency of the source with material with $\mu_0, \rho_0$


## Util command line parameters

```sh
positional arguments:
  source_efficiency_filename -- filename with source efficiency with material 1
  target_efficiency_filename -- filename with target efficiency with material 2

options:
  -h, --help            show this help message and exit
  --path_to_xcom PATH_TO_XCOM, -x PATH_TO_XCOM
                        path to XCOM files with mass attenuation factors
  --start_thick START_THICK, -t START_THICK
                        start thickness in cm to search, it is usually close
                        to real thickness
  --alpha ALPHA         regularization parameter to limit calculated thick

```

## Algo

To find optimum the code uses loss-function L2-loss (least squares):

$$
L(t) = \sum_{i=0}{(eff_1(E_i|t) - eff_2(E_i))^2} + \alpha t
$$
