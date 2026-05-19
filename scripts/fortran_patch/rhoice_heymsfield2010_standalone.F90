!! Standalone Fortran driver for rhoice_heymsfield2010.
!! Kernel transcribed verbatim from rhoice_heymsfield2010.F90:43-99.
!!
!! Input file (whitespace, free-form):
!!   regime  rhoice  rmassmin  rmrat  NBIN
!! Output: real64[2*NBIN]  (rho(1..NBIN), aratelem(1..NBIN))

program rhoice_heymsfield2010_standalone
  implicit none
  integer, parameter :: f = selected_real_kind(15, 307)

  character(len=512) :: in_path, out_path
  character(len=4)   :: regime
  integer            :: unit_in, unit_out, ios, nbin, ibin

  real(kind=f)       :: rhoice, rmassmin, rmrat, a
  real(kind=f), parameter :: b = 2.1_f
  real(kind=f), parameter :: PI = 3.14159265358979_f
  real(kind=f), allocatable :: rho(:), aratelem(:)
  real(kind=f)       :: totalmass, rbin, dmax

  if (command_argument_count() < 2) then
    write(0, '(A)') 'usage: rhoice_heymsfield2010_standalone <in.txt> <out.bin>'
    call exit(1)
  end if
  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)

  open(newunit=unit_in, file=trim(in_path), action='read', status='old', iostat=ios)
  read(unit_in, *, iostat=ios) regime, rhoice, rmassmin, rmrat, nbin
  if (ios /= 0) then; write(0, '(A)') 'malformed header'; call exit(3); end if
  close(unit_in)

  ! Regime → a coefficient. Verbatim from rhoice_heymsfield2010.F90:46-57.
  if (regime == "deep") then
    a = 1.10e-2_f
  else if (regime == "conv") then
    a = 6.33e-3_f
  else if (regime == "cold") then
    a = 5.74e-3_f
  else if (regime == "avg") then
    a = 5.28e-3_f
  else if (regime == "synp") then
    a = 4.22e-3_f
  else if (regime == "warm") then
    a = 3.79e-3_f
  else
    write(0, '(A,A,A)') 'unknown regime: ', trim(regime), ''
    call exit(4)
  end if

  allocate(rho(nbin), aratelem(nbin))
  do ibin = 1, nbin
    totalmass = rmassmin * (rmrat ** (ibin - 1))
    rbin = ((totalmass / a) ** (1._f / b)) / 2._f
    rho(ibin) = totalmass / ((4._f / 3._f) * PI * (rbin ** 3._f))
    rho(ibin) = min(rho(ibin), rhoice)
    dmax = 2._f * rbin
    if (dmax <= 200.e-4_f) then
      aratelem(ibin) = exp(-38._f * dmax)
    else
      aratelem(ibin) = 0.16_f * (dmax ** (-0.27_f))
    end if
  end do

  open(newunit=unit_out, file=trim(out_path), access='stream', form='unformatted', &
       action='write', status='replace', iostat=ios)
  write(unit_out) rho
  write(unit_out) aratelem
  close(unit_out)
  deallocate(rho, aratelem)
end program rhoice_heymsfield2010_standalone
