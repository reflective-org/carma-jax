!! Standalone Fortran driver for hetnucl.
!! Kernel transcribed verbatim from hetnucl.F90:77-159.
!!
!! Input file (whitespace, free-form):
!!   T  p_dyn_cm2  ssi  gc_h2o  gwtmol_h2o  surfctia  pconmax  NBIN
!!   r(1) r(2) ... r(NBIN)
!! Output: real64[NBIN]  rnuclg

program hetnucl_standalone
  implicit none
  integer, parameter :: f = selected_real_kind(15, 307)

  character(len=512) :: in_path, out_path
  integer            :: unit_in, unit_out, ios, nbin, ibin

  real(kind=f)       :: t, p, ssi, gc_h2o, gwtmol_h2o, surfctia, pconmax
  real(kind=f), allocatable :: r(:), rnuclg(:)

  real(kind=f), parameter :: PI       = 3.14159265358979_f
  real(kind=f), parameter :: AVG      = 6.02252e23_f
  real(kind=f), parameter :: BK       = 1.38054e-16_f
  real(kind=f), parameter :: RGAS     = 8.31430e+7_f
  real(kind=f), parameter :: RHO_I    = 0.93_f
  real(kind=f), parameter :: SMALL_PC = 1.0e-50_f
  real(kind=f), parameter :: FEW_PC   = SMALL_PC * 1.0e6_f

  real(kind=f), parameter :: gdes    = 2.9e-13_f
  real(kind=f), parameter :: gsd     = 2.9e-14_f
  real(kind=f), parameter :: zeld    = 0.1_f
  real(kind=f), parameter :: vibfreq = 1.e13_f
  real(kind=f), parameter :: diflen  = 0.1e-7_f
  real(kind=f), parameter :: rmiv    = 0.95_f

  real(kind=f) :: rmw, R_H2O, rnh2o, rlogs, ag, contang
  real(kind=f) :: xh, phih, rath, fv3h, fv4h, fh, delfg, expon

  if (command_argument_count() < 2) then
    write(0, '(A)') 'usage: hetnucl_standalone <in.txt> <out.bin>'; call exit(1)
  end if
  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)

  open(newunit=unit_in, file=trim(in_path), action='read', status='old', iostat=ios)
  read(unit_in, *, iostat=ios) t, p, ssi, gc_h2o, gwtmol_h2o, surfctia, pconmax, nbin
  allocate(r(nbin), rnuclg(nbin))
  read(unit_in, *, iostat=ios) (r(ibin), ibin = 1, nbin)
  close(unit_in)

  rnuclg(:) = 0.0_f
  if (p < 1.e3_f .and. ssi > 0._f .and. pconmax > FEW_PC) then
    rmw = gwtmol_h2o / AVG
    R_H2O = RGAS / gwtmol_h2o
    rnh2o = gc_h2o * R_H2O / BK
    rlogs = log(ssi + 1._f)
    contang = acos(rmiv)

    do ibin = 1, nbin
      ag = 2._f * gwtmol_h2o * surfctia / RGAS / t / RHO_I / rlogs
      xh = r(ibin) / ag
      phih = sqrt(1._f - 2._f * rmiv * xh + xh**2)
      rath = (xh - rmiv) / phih
      fv3h = xh**3 * (2._f - 3._f * rath + rath**3)
      fv4h = 3._f * rmiv * xh**2 * (rath - 1._f)
      if (abs(rath) .gt. 1._f - 1.e-8_f)  fv3h = 0._f
      if (abs(rath) .gt. 1._f - 1.e-10_f) fv4h = 0._f
      fh = 0.5_f * (1._f + ((1._f - rmiv * xh) / phih)**3 + fv3h + fv4h)
      delfg = 4._f * PI * ag**2 * surfctia - 4._f * PI * RHO_I * ag**3 * BK * t * rlogs / 3._f / rmw
      expon = (2._f * gdes - gsd - fh * delfg) / BK / t
      rnuclg(ibin) = min(1.e10_f, zeld * BK * t * diflen * ag * sin(contang) * &
        4._f * PI * r(ibin)**2 * rnh2o**2 / (fh * rmw * vibfreq) * exp(expon))
    end do
  end if

  open(newunit=unit_out, file=trim(out_path), access='stream', form='unformatted', &
       action='write', status='replace', iostat=ios)
  write(unit_out) rnuclg
  close(unit_out)
  deallocate(r, rnuclg)
end program hetnucl_standalone
