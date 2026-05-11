!! Standalone Fortran driver for melticel.
!! Kernel transcribed verbatim from melticel.F90:38-70.
!!
!! Input file (whitespace, free-form):
!!   T  pconmax  NBIN
!! Output: real64[NBIN]  rnuclg  (Fortran stream)

program melticel_standalone
  implicit none
  integer, parameter :: f = selected_real_kind(15, 307)

  character(len=512) :: in_path, out_path
  integer            :: unit_in, unit_out, ios, nbin, ibin
  real(kind=f)       :: t, pconmax
  real(kind=f), allocatable :: rnuclg(:)

  real(kind=f), parameter :: T0       = 273.16_f
  real(kind=f), parameter :: SMALL_PC = 1.0e-50_f
  real(kind=f), parameter :: FEW_PC   = SMALL_PC * 1.0e6_f

  if (command_argument_count() < 2) then
    write(0, '(A)') 'usage: melticel_standalone <in.txt> <out.bin>'
    call exit(1)
  end if
  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)

  open(newunit=unit_in, file=trim(in_path), action='read', status='old', iostat=ios)
  read(unit_in, *, iostat=ios) t, pconmax, nbin
  close(unit_in)
  if (ios /= 0) then; write(0, '(A)') 'malformed header'; call exit(3); end if

  allocate(rnuclg(nbin))
  rnuclg(:) = 0.0_f
  if (pconmax > FEW_PC) then
    if (t > T0) then
      do ibin = 1, nbin
        rnuclg(ibin) = 1.e2_f
      end do
    end if
  end if

  open(newunit=unit_out, file=trim(out_path), access='stream', form='unformatted', &
       action='write', status='replace', iostat=ios)
  if (ios /= 0) then
    write(0, '(A, A)') 'cannot open: ', trim(out_path); call exit(5)
  end if
  write(unit_out) rnuclg
  close(unit_out)
  deallocate(rnuclg)
end program melticel_standalone
