program neural_network
  implicit none
  integer, parameter :: INPUT_SIZE = 2   ! Liczba wejść
  integer, parameter :: HIDDEN_SIZE = 4  ! Liczba neuronów w warstwie ukrytej
  integer, parameter :: OUTPUT_SIZE = 1  ! Liczba wyjść
  real :: w1(INPUT_SIZE, HIDDEN_SIZE)   ! Wagi warstwy wejściowej-ukrytej
  real :: w2(HIDDEN_SIZE, OUTPUT_SIZE)  ! Wagi warstwy ukrytej-wyjściowej
  real :: b1(HIDDEN_SIZE)               ! Bias warstwy ukrytej
  real :: b2(OUTPUT_SIZE)               ! Bias warstwy wyjściowej
  real :: x(INPUT_SIZE)                 ! Wejście
  real :: h(HIDDEN_SIZE)                ! Wyjście warstwy ukrytej
  real :: y(OUTPUT_SIZE)                ! Wyjście sieci
  real :: target(OUTPUT_SIZE)           ! Wartość oczekiwana
  real :: learning_rate = 0.1           ! Współczynnik uczenia
  integer :: i, epoch

  ! Inicjalizacja wag i biasów losowymi wartościami
  call random_seed()
  call random_number(w1)
  call random_number(w2)
  call random_number(b1)
  call random_number(b2)

  ! Przykładowe dane (XOR)
  x = (/ 1.0, 1.0 /)
  target = (/ 0.0 /)

  ! Uczenie sieci (1000 epok)
  do epoch = 1, 1000
    ! Propagacja w przód
    h = matmul(x, w1) + b1
    h = sigmoid(h)
    y = matmul(h, w2) + b2
    y = sigmoid(y)

    ! Obliczenie błędu
    call backpropagation(x, h, y, target, w1, w2, b1, b2, learning_rate)

    ! Wyświetlenie błędu co 100 epok
    if (mod(epoch, 100) == 0) then
      print *, "Epoka:", epoch, "Wyjście:", y, "Błąd:", sum((y - target)**2)
    end if
  end do

contains
  ! Funkcja sigmoidalna
  function sigmoid(x) result(res)
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = 1.0 / (1.0 + exp(-x))
  end function sigmoid

  ! Propagacja wsteczna (uproszczona)
  subroutine backpropagation(x, h, y, target, w1, w2, b1, b2, lr)
    real, intent(in) :: x(:), h(:), y(:), target(:), lr
    real, intent(inout) :: w1(:,:), w2(:,:), b1(:), b2(:)
    real :: delta_out(OUTPUT_SIZE), delta_hidden(HIDDEN_SIZE)
    real :: error(OUTPUT_SIZE)

    ! Błąd wyjścia
    error = y - target
    delta_out = error * y * (1.0 - y)

    ! Aktualizacja wag i biasów warstwy wyjściowej
    w2 = w2 - lr * spread(h, 2, OUTPUT_SIZE) * spread(delta_out, 1, HIDDEN_SIZE)
    b2 = b2 - lr * delta_out

    ! Błąd warstwy ukrytej
    delta_hidden = matmul(delta_out, transpose(w2)) * h * (1.0 - h)

    ! Aktualizacja wag i biasów warstwy ukrytej
    w1 = w1 - lr * spread(x, 2, HIDDEN_SIZE) * spread(delta_hidden, 1, INPUT_SIZE)
    b1 = b1 - lr * delta_hidden
  end subroutine backpropagation
end program neural_network
