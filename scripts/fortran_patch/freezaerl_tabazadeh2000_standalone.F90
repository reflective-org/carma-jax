!! Standalone Fortran driver for freezaerl_tabazadeh2000.
!!
!! Reads scenario inputs from a text file (argv[1]) and writes the
!! per-bin nucleation loss rate (NBIN real64 values, Fortran
!! unformatted-stream) to argv[2]. Kernel is transcribed verbatim from
!! `original-carma/CARMA/source/base/freezaerl_tabazadeh2000.F90:75-300`.
!!
!! Input file format (whitespace, free-form):
!!   T  ssi  ssl  akelvin  rhosol  gwtmol  pconmax  NBIN
!!   r(1)   r(2)   ... r(NBIN)
!!   vol(1) vol(2) ... vol(NBIN)
!!
!! Output: real64[NBIN]   rnuclg   (Fortran stream)

program freezaerl_tabazadeh2000_standalone
  implicit none
  integer, parameter :: f = selected_real_kind(15, 307)

  character(len=512) :: in_path, out_path
  integer            :: unit_in, unit_out, ios, nbin, ibin

  real(kind=f)       :: t, ssi_in, ssl_in, akelvin, rhosol, gwtmol, pconmax
  real(kind=f), allocatable :: r(:), vol(:), rnuclg(:)

  ! Constants from carma_constants_mod.F90 / carma_globaer.h
  real(kind=f), parameter :: T0       = 273.16_f
  real(kind=f), parameter :: PI       = 3.14159265358979_f
  real(kind=f), parameter :: BK       = 1.38054e-16_f
  real(kind=f), parameter :: RGAS     = 8.31430e+7_f
  real(kind=f), parameter :: RHO_W    = 1.0_f
  real(kind=f), parameter :: RHO_I    = 0.93_f
  real(kind=f), parameter :: SMALL_PC = 1.0e-50_f
  real(kind=f), parameter :: FEW_PC   = SMALL_PC * 1.0e6_f
  real(kind=f), parameter :: SIFREEZE = 0.3_f
  real(kind=f), parameter :: PRENUC   = 2.075e33_f * RHO_W / RHO_I
  real(kind=f), parameter :: ONE      = 1.0_f

  ! Scratch
  real(kind=f) :: ssl, act, fkelv
  real(kind=f) :: contl, conth, h2so4m, wt, vrat
  real(kind=f) :: c1c, c2c, c3c, c4c
  real(kind=f) :: A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10
  real(kind=f) :: wtfrac, den
  real(kind=f) :: diffact
  real(kind=f) :: c0, cc1, cc2, cc3, cc4, cc5
  real(kind=f) :: d0, d1, d2, d3, d4, d5
  real(kind=f) :: e0, e1, e2, e3, e4, e5
  real(kind=f) :: S260, S220, S180
  real(kind=f) :: sigma, sigsula, sigicea, sigsulice
  real(kind=f) :: rhoibar, rlhbar
  real(kind=f) :: ag, delfg, expon

  if (command_argument_count() < 2) then
    write(0, '(A)') 'usage: freezaerl_tabazadeh2000_standalone <in.txt> <out.bin>'
    call exit(1)
  end if
  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)

  open(newunit=unit_in, file=trim(in_path), action='read', status='old', iostat=ios)
  if (ios /= 0) then
    write(0, '(A, A)') 'cannot open input: ', trim(in_path); call exit(2)
  end if
  read(unit_in, *, iostat=ios) t, ssi_in, ssl_in, akelvin, rhosol, gwtmol, pconmax, nbin
  if (ios /= 0) then; write(0, '(A)') 'malformed header'; call exit(3); end if
  allocate(r(nbin), vol(nbin), rnuclg(nbin))
  read(unit_in, *, iostat=ios) (r(ibin),   ibin = 1, nbin)
  read(unit_in, *, iostat=ios) (vol(ibin), ibin = 1, nbin)
  close(unit_in)
  if (ios /= 0) then; write(0, '(A)') 'malformed bin arrays'; call exit(4); end if

  rnuclg(:) = 0.0_f
  sigicea = 105._f

  if (pconmax > FEW_PC .and. ssi_in > SIFREEZE) then
    rhoibar = ( 0.916_f * (t-T0) - &
                1.75e-4_f/2._f * ((t-T0)**2) - &
                5.e-7_f * ((t-T0)**3)/3._f ) / (t-T0)
    rlhbar = ( 79.7_f * (t-T0) + &
               0.485_f/2._f * (t-T0)**2 - &
               2.5e-3_f/3._f * (t-T0)**3 ) / (t-T0) * 4.186e7_f * 18._f

    ssl = max(-1.0_f, min(0._f, ssl_in))
    act = min(1.0_f, ssl + 1._f)

    ! Activation energy (Koop lab data) — picked once per scenario, not per bin.
    if (t .GT. 220._f) then
      A0 = 104525.93058_f;       A1 = -1103.7644651_f;      A2 = 1.070332702_f
      A3 = 0.017386254322_f;     A4 = -1.5506854268e-06_f;  A5 = -3.2661912497e-07_f
      A6 = 6.467954459e-10_f
    else
      A0 = -17459.516183_f;      A1 = 458.45827551_f;       A2 = -4.8492831317_f
      A3 = 0.026003658878_f;     A4 = -7.1991577798e-05_f;  A5 = 8.9049094618e-08_f
      A6 = -2.4932257419e-11_f
    end if
    diffact = (A0 + A1*t + A2*t**2 + A3*t**3 + A4*t**4 + A5*t**5 + A6*t**6) * 1.0e-13_f

    do ibin = 1, nbin
      ! Kelvin-corrected activity (per bin).
      fkelv = exp(akelvin / r(ibin))
      act = min(1.0_f, ssl + 1._f) / fkelv

      if (act .LT. 0.05_f) then
        contl = 12.37208932_f  * (act**(-0.16125516114_f)) - 30.490657554_f * act - 2.1133114241_f
        conth = 13.455394705_f * (act**(-0.1921312255_f))  - 34.285174604_f * act - 1.7620073078_f
      else if (act .LE. 0.85_f) then
        contl = 11.820654354_f * (act**(-0.20786404244_f)) - 4.807306373_f  * act - 5.1727540348_f
        conth = 12.891938068_f * (act**(-0.23233847708_f)) - 6.4261237757_f * act - 4.9005471319_f
      else
        contl = -180.06541028_f * (act**(-0.38601102592_f)) - 93.317846778_f * act + 273.88132245_f
        conth = -176.95814097_f * (act**(-0.36257048154_f)) - 90.469744201_f * act + 267.45509988_f
      end if

      h2so4m = contl + ((conth - contl) * (t - 190._f) / 70._f)
      wt = (98.0_f * h2so4m) / (1000._f + 98._f * h2so4m)
      wt = 100._f * wt

      if (wt <= 0._f) cycle           ! parameterisation breakdown — skip

      vrat = rhosol/RHO_W * ((100._f - wt)/wt) + 1._f

      ! Sulfate solution density (Myhre 1998) — recomputed only because
      ! the surface-tension polynomials need WT-derived state. We don't
      ! actually use `den` in the rate but Fortran evaluates it; keep for
      ! parity in case future wt-clamping branches read it.
      wtfrac = WT/100._f
      c1c = t - 273.15_f
      c2c = c1c**2; c3c = c1c**3; c4c = c1c**4
      A0 = 999.8426_f + 334.5402e-4_f*c1c - 569.1304e-5_f*c2c
      A1 = 547.2659_f - 530.0445e-2_f*c1c + 118.7671e-4_f*c2c + 599.0008e-6_f*c3c
      A2 = 526.295e+1_f + 372.0445e-1_f*c1c + 120.1909e-3_f*c2c - 414.8594e-5_f*c3c + 119.7973e-7_f*c4c
      A3 = -621.3958e+2_f - 287.7670_f*c1c - 406.4638e-3_f*c2c + 111.9488e-4_f*c3c + 360.7768e-7_f*c4c
      A4 = 409.0293e+3_f + 127.0854e+1_f*c1c + 326.9710e-3_f*c2c - 137.7435e-4_f*c3c - 263.3585e-7_f*c4c
      A5 = -159.6989e+4_f - 306.2836e+1_f*c1c + 136.6499e-3_f*c2c + 637.3031e-5_f*c3c
      A6 = 385.7411e+4_f + 408.3717e+1_f*c1c - 192.7785e-3_f*c2c
      A7 = -580.8064e+4_f - 284.4401e+1_f*c1c
      A8 = 530.1976e+4_f + 809.1053_f*c1c
      A9 = -268.2616e+4_f
      A10 = 576.4288e+3_f
      den = A0 + wtfrac*A1 + wtfrac**2*A2 + wtfrac**3*A3 + wtfrac**4*A4 &
          + wtfrac**5*A5 + wtfrac**6*A6 + wtfrac**7*A7 &
          + wtfrac**8*A8 + wtfrac**9*A9 + wtfrac**10*A10

      ! Surface tensions (S260, S220, S180).
      c0  = 77.40682664_f;  cc1 = -0.006963123274_f
      cc2 = -0.009682499074_f; cc3 = 0.00088797988_f
      cc4 = -2.384669516e-05_f; cc5 = 2.095358048e-07_f
      S260 = c0 + cc1*wt + cc2*wt**2 + cc3*wt**3 + cc4*wt**4 + cc5*wt**5
      d0 = 82.01197792_f; d1 = 0.5312072092_f
      d2 = -0.1050692123_f; d3 = 0.005415260617_f
      d4 = -0.0001145573827_f; d5 = 8.969257061e-07_f
      S220 = d0 + d1*wt + d2*wt**2 + d3*wt**3 + d4*wt**4 + d5*wt**5
      e0 = 85.75507114_f; e1 = 0.09541966318_f
      e2 = -0.1103647657_f; e3 = 0.007485866933_f
      e4 = -0.0001912224154_f; e5 = 1.736789787e-06_f
      S180 = e0 + e1*wt + e2*wt**2 + e3*wt**3 + e4*wt**4 + e5*wt**5

      if (t .GE. 220._f) then
        sigma = S260 + ((260._f - t) * (S220 - S260)) / 40._f
      else
        sigma = S220 + ((220._f - t) * (S180 - S220)) / 40._f
      end if
      sigsula = sigma
      sigsulice = abs(sigsula - sigicea)

      ag = 2._f * gwtmol * sigsulice / &
           ( rlhbar * rhoibar * log(T0/t) + &
             rhoibar * RGAS * 0.5_f * (T0+t) * log(ssl + 1._f) )
      if (ag .LT. 0._f) ag = 1.e10_f

      delfg = 4._f/3._f * PI * sigsulice * (ag**2)

      expon = (-diffact - delfg) / BK / t
      expon = max(-100._f * ONE, expon)
      rnuclg(ibin) = PRENUC * sqrt(sigsulice * t) * vrat * vol(ibin) * exp(expon)
      if (rnuclg(ibin) < 0._f) rnuclg(ibin) = 0._f
    end do
  end if

  open(newunit=unit_out, file=trim(out_path), access='stream', form='unformatted', &
       action='write', status='replace', iostat=ios)
  if (ios /= 0) then
    write(0, '(A, A)') 'cannot open output: ', trim(out_path); call exit(5)
  end if
  write(unit_out) rnuclg
  close(unit_out)

  deallocate(r, vol, rnuclg)
end program freezaerl_tabazadeh2000_standalone
