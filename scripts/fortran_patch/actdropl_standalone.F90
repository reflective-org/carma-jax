!! Standalone Fortran driver for actdropl.
!! Kernel transcribed verbatim from actdropl.F90:44-102.
!!
!! Input file (whitespace, free-form):
!!   T  supsatl  pconmax  NBIN
!!   scrit(1)         ... scrit(NBIN)
!!   pc(1)            ... pc(NBIN)
!!   target_evap(1)   ... target_evap(NBIN)   (0 or 1; 1 = target evaporating)
!! Output: real64[NBIN]  rnuclg

program actdropl_standalone
  implicit none
  integer, parameter :: f = selected_real_kind(15, 307)

  character(len=512) :: in_path, out_path
  integer            :: unit_in, unit_out, ios, nbin, ibin

  real(kind=f)       :: t, supsatl, pconmax
  real(kind=f), allocatable :: scrit(:), pc(:), target_evap(:), rnuclg(:)

  real(kind=f), parameter :: T0       = 273.16_f
  real(kind=f), parameter :: SMALL_PC = 1.0e-50_f
  real(kind=f), parameter :: FEW_PC   = SMALL_PC * 1.0e6_f
  logical :: evapfrom_nucto

  if (command_argument_count() < 2) then
    write(0, '(A)') 'usage: actdropl_standalone <in.txt> <out.bin>'
    call exit(1)
  end if
  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)

  open(newunit=unit_in, file=trim(in_path), action='read', status='old', iostat=ios)
  read(unit_in, *, iostat=ios) t, supsatl, pconmax, nbin
  if (ios /= 0) then; write(0, '(A)') 'malformed header'; call exit(3); end if
  allocate(scrit(nbin), pc(nbin), target_evap(nbin), rnuclg(nbin))
  read(unit_in, *, iostat=ios) (scrit(ibin),       ibin = 1, nbin)
  read(unit_in, *, iostat=ios) (pc(ibin),          ibin = 1, nbin)
  read(unit_in, *, iostat=ios) (target_evap(ibin), ibin = 1, nbin)
  close(unit_in)

  rnuclg(:) = 0.0_f
  if (t >= (T0 - 40._f) .and. pconmax > FEW_PC) then
    do ibin = 1, nbin
      evapfrom_nucto = (target_evap(ibin) > 0._f)
      if (supsatl > scrit(ibin) .and. .not. evapfrom_nucto .and. pc(ibin) > SMALL_PC) then
        rnuclg(ibin) = 1.e3_f
      end if
    end do
  end if

  open(newunit=unit_out, file=trim(out_path), access='stream', form='unformatted', &
       action='write', status='replace', iostat=ios)
  write(unit_out) rnuclg
  close(unit_out)
  deallocate(scrit, pc, target_evap, rnuclg)
end program actdropl_standalone
