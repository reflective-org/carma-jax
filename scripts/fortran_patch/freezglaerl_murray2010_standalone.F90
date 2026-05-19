!! Standalone Fortran driver for freezglaerl_murray2010.
!! Kernel transcribed verbatim from freezglaerl_murray2010.F90:62-132.
!!
!! Murray et al. 2010 glassy-aerosol heterogeneous freezing: the rate
!! is uniform across all bins (it's a fraction-of-aerosols-nucleated
!! formula, not bin-radius-dependent).
!!
!! Input file (whitespace, free-form):
!!   T  ssi  ssi_old  pconmax  dtime  NBIN
!! Output: real64[NBIN]  rnuclg

program freezglaerl_murray2010_standalone
  implicit none
  integer, parameter :: f = selected_real_kind(15, 307)

  character(len=512) :: in_path, out_path
  integer            :: unit_in, unit_out, ios, nbin, ibin

  real(kind=f)       :: t, ssi, ssiold, pconmax, dtime
  real(kind=f), allocatable :: rnuclg(:)

  ! Murray fit constants
  real(kind=f), parameter :: kice1   = 7.7211e-5_f
  real(kind=f), parameter :: kice2   = 9.2688e-3_f
  real(kind=f), parameter :: ssmin   = 0.21_f
  real(kind=f), parameter :: ssmax   = 0.7_f
  real(kind=f), parameter :: tglass  = 212._f
  real(kind=f), parameter :: fglass  = 0.5_f
  real(kind=f), parameter :: SMALL_PC = 1.0e-50_f
  real(kind=f), parameter :: FEW_PC   = SMALL_PC * 1.0e6_f

  real(kind=f) :: dfice, rate

  if (command_argument_count() < 2) then
    write(0, '(A)') 'usage: freezglaerl_murray2010_standalone <in.txt> <out.bin>'
    call exit(1)
  end if
  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)

  open(newunit=unit_in, file=trim(in_path), action='read', status='old', iostat=ios)
  read(unit_in, *, iostat=ios) t, ssi, ssiold, pconmax, dtime, nbin
  close(unit_in)
  if (ios /= 0) then; write(0, '(A)') 'malformed header'; call exit(3); end if

  allocate(rnuclg(nbin))
  rnuclg(:) = 0.0_f

  if (t <= tglass .and. pconmax > FEW_PC .and. &
      ssi >= ssmin .and. ssi > ssiold) then
    dfice = kice1 * (1._f + min(ssmax, ssi)) * 100._f - kice2
    if (ssiold >= ssmin) then
      dfice = dfice - (kice1 * (1._f + min(ssmax, ssiold)) * 100._f - kice2)
    end if
    rate = fglass * dfice / dtime
    do ibin = 1, nbin
      rnuclg(ibin) = rate
    end do
  end if

  open(newunit=unit_out, file=trim(out_path), access='stream', form='unformatted', &
       action='write', status='replace', iostat=ios)
  write(unit_out) rnuclg
  close(unit_out)
  deallocate(rnuclg)
end program freezglaerl_murray2010_standalone
