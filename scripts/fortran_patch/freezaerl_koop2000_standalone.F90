!! Standalone Fortran driver for freezaerl_koop2000.
!!
!! Kernel transcribed verbatim from
!! `original-carma/CARMA/source/base/freezaerl_koop2000.F90:79-205`.
!!
!! Input file format (whitespace, free-form):
!!   T  p_dyn_cm2  ssi  ssl  akelvin  rhosol  pconmax  NBIN
!!   r(1)   r(2)   ... r(NBIN)
!!   vol(1) vol(2) ... vol(NBIN)
!!
!! Output: real64[NBIN]   rnuclg   (Fortran stream)

program freezaerl_koop2000_standalone
  implicit none
  integer, parameter :: f = selected_real_kind(15, 307)

  character(len=512) :: in_path, out_path
  integer            :: unit_in, unit_out, ios, nbin, ibin

  real(kind=f)       :: t, p, ssi_in, ssl_in, akelvin
  real(kind=f)       :: rhosol, pconmax
  real(kind=f), allocatable :: r(:), vol(:), rnuclg(:)

  ! Constants from carma_constants_mod.F90
  real(kind=f), parameter :: PI       = 3.14159265358979_f
  real(kind=f), parameter :: BK       = 1.38054e-16_f
  real(kind=f), parameter :: RGAS     = 8.31430e+7_f
  real(kind=f), parameter :: RHO_W    = 1.0_f
  real(kind=f), parameter :: RHO_I    = 0.93_f
  real(kind=f), parameter :: SMALL_PC = 1.0e-50_f
  real(kind=f), parameter :: FEW_PC   = SMALL_PC * 1.0e6_f
  real(kind=f), parameter :: SIFREEZE = 0.3_f
  real(kind=f), parameter :: PRENUC   = 2.075e33_f * RHO_W / RHO_I
  real(kind=f), parameter :: KT0      = 1.6e0_f
  real(kind=f), parameter :: DKT0DP   = -8.8e0_f
  real(kind=f), parameter :: KTI      = 0.22e0_f
  real(kind=f), parameter :: DKTIDP   = -0.17e0_f
  real(kind=f), parameter :: POWMAX   = 706.0_f
  real(kind=f), parameter :: ONE      = 1.0_f

  ! Scratch
  real(kind=f) :: ssi, ssl, fkelv
  real(kind=f) :: td, rlnt, dmy, rsi, awi
  real(kind=f) :: vw0, vi
  real(kind=f) :: pp, pp2, pp3, riv
  real(kind=f) :: aw, daw, rlogj, rjj
  real(kind=f) :: contl, conth, h2so4m, wt, volrat

  if (command_argument_count() < 2) then
    write(0, '(A)') 'usage: freezaerl_koop2000_standalone <in.txt> <out.bin>'
    call exit(1)
  end if
  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)

  open(newunit=unit_in, file=trim(in_path), action='read', status='old', iostat=ios)
  if (ios /= 0) then
    write(0, '(A, A)') 'cannot open input: ', trim(in_path); call exit(2)
  end if
  read(unit_in, *, iostat=ios) t, p, ssi_in, ssl_in, akelvin, rhosol, pconmax, nbin
  if (ios /= 0) then; write(0, '(A)') 'malformed header'; call exit(3); end if
  allocate(r(nbin), vol(nbin), rnuclg(nbin))
  read(unit_in, *, iostat=ios) (r(ibin),   ibin = 1, nbin)
  read(unit_in, *, iostat=ios) (vol(ibin), ibin = 1, nbin)
  close(unit_in)
  if (ios /= 0) then; write(0, '(A)') 'malformed bin arrays'; call exit(4); end if

  rnuclg(:) = 0.0_f

  ! T <= 240 gate
  if (t <= 240.0_f .and. pconmax > FEW_PC) then
    do ibin = 1, nbin
      ssi = ssi_in
      ssl = ssl_in
      if (ssi > SIFREEZE) then

        ! Koop et al. nucleation rate parameterization
        td   = t
        rlnt = log(td)
        dmy  = 210368._f + 131.438_f * td - (3.32373e6_f / td) - 41729.1_f * rlnt
        rsi  = RGAS / 1.e7_f
        awi  = exp(dmy / (rsi * td))

        vw0  = -230.76_f - 0.1478_f * td + (4099.2_f / td) + 48.8341_f * rlnt
        vi   = 19.43_f - 2.2e-3_f * td + 1.08e-5_f * td * td

        pp   = 1.e-10_f * p
        pp2  = pp * pp * 0.5_f
        pp3  = pp2 * pp / 3._f
        riv  = vw0 * (pp - KT0 * pp2 - DKT0DP * pp3) - &
               vi  * (pp - KTI * pp2 - DKTIDP * pp3)
        riv  = riv * 1.e3_f

        ssl = max(-1.0_f, min(0._f, ssl))
        aw  = 1._f + ssl
        fkelv = exp(akelvin / r(ibin))
        aw  = aw / fkelv

        daw   = aw * exp(riv / (rsi * td)) - awi
        daw   = min(0.34_f, max(daw, 0.26_f))

        rlogj = ((29180._f * daw - 26924._f) * daw + 8502._f) * daw - 906.7_f
        rlogj = min(rlogj, POWMAX * 0.3_f)
        rjj   = 10._f ** rlogj

        if (aw < 0.05_f) then
          contl =  12.37208932_f  * (aw**(-0.16125516114_f)) - 30.490657554_f * aw - 2.1133114241_f
          conth =  13.455394705_f * (aw**(-0.1921312255_f))  - 34.285174604_f * aw - 1.7620073078_f
        else if (aw <= 0.85_f) then
          contl =  11.820654354_f * (aw**(-0.20786404244_f)) - 4.807306373_f  * aw - 5.1727540348_f
          conth =  12.891938068_f * (aw**(-0.23233847708_f)) - 6.4261237757_f * aw - 4.9005471319_f
        else
          contl = -180.06541028_f * (aw**(-0.38601102592_f)) - 93.317846778_f * aw + 273.88132245_f
          conth = -176.95814097_f * (aw**(-0.36257048154_f)) - 90.469744201_f * aw + 267.45509988_f
        end if

        h2so4m = contl + ((conth - contl) * (t - 190._f) / 70._f)
        wt = (98.0_f * h2so4m) / (1000._f + 98._f * h2so4m)
        wt = max(0._f, min(1._f, wt))
        wt = 100._f * wt

        if (wt <= 0._f) then
          volrat = 1.e10_f
        else
          volrat = rhosol / RHO_W * ((100._f - wt) / wt) + 1._f
        end if

        rnuclg(ibin) = rjj * volrat * vol(ibin)
      end if
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
end program freezaerl_koop2000_standalone
