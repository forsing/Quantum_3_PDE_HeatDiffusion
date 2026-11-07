# Quantum_3_PDE_HeatDiffusion 
# PDE — Heat/Diffusion



import csv
import itertools
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

from qiskit.quantum_info import DensityMatrix, Operator

random.seed(39)
np.random.seed(39)


FILES = (
    Path("/data/loto7_4716_k89.csv"),
    Path("/data/loto7_4716_k89_loto_2980.csv"),
    Path("/data/loto7_4716_k89_loto_plus_1736.csv"),
)


TITLE = "Quantum PDE Diffusion"
N, K, DIM = 39, 7, 64
REGULARIZATIONS = (0.001, 0.01)
MAX_ITERATIONS = 200
STEPS = 16
DT = 1.0 / STEPS

IDENTITY = np.eye(N)
SHIFT = np.roll(IDENTITY, 1, axis=0)
LAPLACIAN = SHIFT + SHIFT.T - 2.0 * IDENTITY

# Jednacina:
# dp/dt = kappa * Laplacian(p) + rate * (source - p).
#
# Prostorna mreza ima 39 tacaka i periodicne granice.
# Source je naucena normalizovana raspodela.
# Izvorni clan cuva ukupnu verovatnocu.
#
# FTCS korak:
# A = stay*I + kappa*dt*(S + S.T) + rate*dt*source*1.T.
#
# kappa i rate su u (0,3).
# Zato su sve tezine jednog koraka nenegativne.
# Kvantni prikaz je mesavina pomeranja i resetovanja izvora.


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            try:
                values = tuple(int(value) for value in row)
            except ValueError as error:
                raise ValueError(f"{path.name}: red {line}.") from error
            check(
                len(values) == K
                and len(set(values)) == K
                and min(values) >= 1
                and max(values) <= N
                and values == tuple(sorted(values)),
                f"{path.name}: nelegalan red {line}.",
            )
            rows.append(values)
    check(len(rows) >= 5, f"{path.name}: nedovoljno kola.")
    return np.asarray(rows, dtype=int) - 1


def initial():
    return np.concatenate((np.full(2, -1.0), np.zeros(N)))


def transition(theta):
    kappa, rate = 3.0 * expit(theta[:2])
    source = softmax(theta[2:])
    stay = 1.0 - DT * (2.0 * kappa + rate)
    step = (
        stay * IDENTITY
        + kappa * DT * (SHIFT + SHIFT.T)
        + rate * DT * source[:, None]
    )
    powers = [IDENTITY]
    for _ in range(STEPS):
        powers.append(step @ powers[-1])
    probabilities = np.zeros((DIM, N))
    probabilities[:N] = powers[-1]
    return probabilities, (step, powers, source, kappa, rate)


def data_loss(probabilities, counts):
    check(
        probabilities.shape == (DIM, N)
        and np.all(np.isfinite(probabilities))
        and np.all(probabilities >= 0)
        and np.allclose(
            probabilities.sum(axis=0), 1.0, atol=1e-8, rtol=0
        ),
        "Neispravna uslovna raspodela.",
    )
    total = counts.sum()
    previous_counts = counts.sum(axis=0)
    loss = (
        np.sum(probabilities**2 * previous_counts)
        - 2.0 * np.sum(counts * probabilities)
    ) / total + 1.0
    derivative = 2.0 * (
        probabilities * previous_counts - counts
    ) / total
    return float(loss), derivative


def objective(theta, counts, weight):
    probabilities, auxiliary = transition(theta)
    step, powers, source, kappa, rate = auxiliary
    loss, derivative = data_loss(probabilities, counts)

    adjoint = derivative[:N]
    step_gradient = np.zeros((N, N))
    for previous in reversed(powers[:-1]):
        step_gradient += adjoint @ previous.T
        adjoint = step.T @ adjoint

    sigmoid = expit(theta[:2])
    rate_derivatives = 3.0 * sigmoid * (1.0 - sigmoid)

    diffusion_gradient = DT * np.sum(step_gradient * LAPLACIAN)
    relaxation_gradient = DT * np.sum(
        step_gradient * (source[:, None] - IDENTITY)
    )
    source_gradient = rate * DT * step_gradient.sum(axis=1)
    logits_gradient = source * (
        source_gradient - np.dot(source, source_gradient)
    )

    gradient = np.concatenate((
        np.array([
            diffusion_gradient * rate_derivatives[0],
            relaxation_gradient * rate_derivatives[1],
        ]),
        logits_gradient,
    ))
    return (
        loss + weight * np.dot(theta, theta),
        gradient + 2.0 * weight * theta,
    )


def joint(probabilities):
    slots, states = probabilities.shape
    indices = np.arange(states)
    increasing = indices[:, None] < indices[None, :]
    allowed = (
        (indices[None, :] >= np.arange(slots)[:, None])
        & (
            indices[None, :]
            <= states - slots + np.arange(slots)[:, None]
        )
    )
    logs = np.full(probabilities.shape, -np.inf)
    np.log(probabilities, out=logs, where=probabilities > 0)
    logs[~allowed] = -np.inf
    forward = np.full_like(logs, -np.inf)
    backward = np.full_like(logs, -np.inf)
    forward[0] = logs[0]
    values = logs[0].copy()
    parents = np.zeros((slots, states), dtype=int)

    for slot in range(1, slots):
        forward[slot] = logs[slot] + logsumexp(
            np.where(increasing, forward[slot - 1, :, None], -np.inf),
            axis=0,
        )
        candidates = np.where(increasing, values[:, None], -np.inf)
        parents[slot] = candidates.argmax(axis=0)
        values = logs[slot] + candidates[
            parents[slot], np.arange(states)
        ]

    log_partition = float(logsumexp(forward[-1]))
    check(np.isfinite(log_partition), "Nema legalne mase.")
    path = np.zeros(slots, dtype=int)
    path[-1] = int(values.argmax())
    for slot in range(slots - 1, 0, -1):
        path[slot - 1] = parents[slot, path[slot]]

    backward[-1] = 0.0
    for slot in range(slots - 2, -1, -1):
        backward[slot] = logsumexp(
            np.where(
                increasing,
                (logs[slot + 1] + backward[slot + 1])[None, :],
                -np.inf,
            ),
            axis=1,
        )
    marginals = np.exp(forward + backward - log_partition)
    check(
        np.allclose(marginals.sum(axis=1), 1.0, atol=1e-8, rtol=0),
        "Marginalne raspodele nisu normalizovane.",
    )
    return path, marginals, log_partition, logs


def counts_for(data, end, slot):
    counts = np.zeros((DIM, N))
    np.add.at(
        counts,
        (data[1:end + 1, slot], data[:end, slot]),
        1.0,
    )
    return counts


def fit(data, end, weight):
    parameters = []
    for slot in range(K):
        result = minimize(
            objective,
            initial(),
            args=(counts_for(data, end, slot), weight),
            jac=True,
            method="CG",
            options={"maxiter": MAX_ITERATIONS, "gtol": 1e-6},
        )
        check(
            np.isfinite(result.fun) and np.all(np.isfinite(result.x)),
            "Obuka nije vratila konacne parametre.",
        )
        print("Registar:", slot + 1, "status:", result.message, flush=True)
        parameters.append(result.x)
    return parameters


def evaluate(parameters, data, times):
    matrices = [transition(theta)[0] for theta in parameters]
    losses, hits = [], []
    for time in times:
        probabilities = np.array([
            matrices[slot][:N, data[time - 1, slot]]
            for slot in range(K)
        ])
        path, _, log_partition, logs = joint(probabilities)
        loss = log_partition - logs[
            np.arange(K), data[time]
        ].sum()
        check(not np.isnan(loss) and loss >= -1e-8, "Neispravan NLL.")
        losses.append(loss)
        hits.append(len(set(path) & set(data[time])))
    return float(np.mean(losses)), np.asarray(hits, dtype=int)


def verify_quantum(theta, previous):
    probabilities, auxiliary = transition(theta)
    _, _, source, kappa, rate = auxiliary
    shift = np.eye(DIM, dtype=complex)
    shift[:N, :N] = SHIFT

    forward_operator = Operator(shift)
    backward_operator = Operator(shift.conj().T)

    source_population = np.zeros(DIM)
    source_population[:N] = source
    source_density = np.diag(source_population)

    density = DensityMatrix.from_int(previous, DIM)
    stay = 1.0 - DT * (2.0 * kappa + rate)

    for _ in range(STEPS):
        forward_density = density.evolve(forward_operator).data
        backward_density = density.evolve(backward_operator).data
        density = DensityMatrix(
            stay * density.data
            + kappa * DT * (forward_density + backward_density)
            + rate * DT * source_density
        )

    check(
        density.is_valid()
        and np.allclose(
            density.probabilities(),
            probabilities[:, previous],
            atol=2e-9,
            rtol=0,
        ),
        "Qiskit kanal i FTCS difuzija nisu usaglaseni.",
    )


def self_check(data):
    counts = counts_for(data, min(8, len(data) - 1), 0)
    theta = initial() + 0.03 * np.sin(np.arange(len(initial())) + 1)
    _, gradient = objective(theta, counts, 0.001)
    for index in range(len(theta)):
        offset = np.zeros_like(theta)
        offset[index] = 1e-6
        numerical = (
            objective(theta + offset, counts, 0.001)[0]
            - objective(theta - offset, counts, 0.001)[0]
        ) / 2e-6
        check(
            np.isclose(numerical, gradient[index], atol=2e-5, rtol=2e-4),
            f"Gradijent nije potvrdjen: {index}.",
        )
    toy = softmax(np.sin(np.arange(15).reshape(3, 5)), axis=1)
    path, _, log_partition, logs = joint(toy)
    scores = [
        sum(logs[j, combination[j]] for j in range(3))
        for combination in itertools.combinations(range(5), 3)
    ]
    check(
        np.isclose(log_partition, logsumexp(scores), atol=1e-10)
        and np.isclose(logs[np.arange(3), path].sum(), max(scores)),
        "Provera globalne raspodele nije prosla.",
    )
    print("Matematicke provere su prosle.")


def main():
    print(TITLE)
    print("Difuzija, konačne razlike i izvor koji cuva verovatnocu.")
    print("Periodicne granice. Broj FTCS koraka:", STEPS)
    datasets = [(path, load(path)) for path in FILES]
    self_check(datasets[0][1])
    results = []

    for path, data in datasets:
        total = len(data) - 1
        train_end = int(0.70 * total)
        validation_end = int(0.85 * total)
        check(
            0 < train_end < validation_end < total,
            "Neispravna hronoloska podela.",
        )
        print()
        print(path.name, "redova:", len(data))
        best_weight, best_loss = None, np.inf
        for weight in REGULARIZATIONS:
            parameters = fit(data, train_end, weight)
            loss, _ = evaluate(
                parameters, data, range(train_end + 1, validation_end + 1)
            )
            print("Regularizacija:", weight, "validacioni NLL:", loss)
            if np.isfinite(loss) and loss < best_loss:
                best_weight, best_loss = weight, loss
        check(best_weight is not None, "Nema konacnog validacionog NLL.")

        test_parameters = fit(data, validation_end, best_weight)
        test_loss, hits = evaluate(
            test_parameters, data, range(validation_end + 1, len(data))
        )
        print("Test kola:", len(hits))
        print("Test NLL:", test_loss)
        print("Test srednji broj pogodaka:", float(hits.mean()))

        parameters = fit(data, total, best_weight)
        raw = []
        for slot, theta in enumerate(parameters):
            previous = int(data[-1, slot])
            verify_quantum(theta, previous)
            raw.append(transition(theta)[0][:N, previous])
            print(
                "Registar:", slot + 1,
                "difuzija i relaksacija:", 3.0 * expit(theta[:2]),
            )
        raw = np.asarray(raw)
        prediction, marginals, log_partition, _ = joint(raw)
        print("Log legalne mase:", log_partition)
        if np.array_equal(prediction, data[-1]):
            print("PROVERA: NEXT ponavlja poslednje kolo.")
        run = 1
        for difference in np.diff(prediction):
            run = run + 1 if difference == 1 else 1
            if run >= 5:
                print("PROVERA: najmanje pet uzastopnih brojeva.")
                break
        results.append((path.name, prediction + 1, marginals, hits, raw))

    for first, second in itertools.combinations(results, 2):
        if np.array_equal(first[1], second[1]):
            print("ISTI NEXT:", first[0], second[0])
            print("Razlika raspodela:", float(np.max(np.abs(first[4] - second[4]))))

    figure, axes = plt.subplots(3, 2, figsize=(12, 10))
    for row, result in enumerate(results):
        name, prediction, marginals, hits, _ = result
        print(name)
        print("NEXT:", " ".join(map(str, prediction)))
        axes[row, 0].imshow(marginals, aspect="auto")
        axes[row, 0].set_title(name)
        axes[row, 0].set_xlabel("Broj - indeks 0 predstavlja broj 1")
        axes[row, 1].bar(np.arange(8), np.bincount(hits, minlength=8))
        axes[row, 1].set_xlabel("Pogodaka na testu")
    figure.suptitle(TITLE)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()



"""
Quantum PDE Diffusion
Difuzija, konačne razlike i izvor koji cuva verovatnocu.
Periodicne granice. Broj FTCS koraka: 16
Matematicke provere su prosle.

loto7_4716_k89.csv redova: 4716
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.001 validacioni NLL: 17.602011750545152
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.01 validacioni NLL: 17.648183784239624
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Test kola: 708
Test NLL: 17.54490153689806
Test srednji broj pogodaka: 1.2330508474576272
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Registar: 1 difuzija i relaksacija: [0.9153552  1.97340315]
Registar: 2 difuzija i relaksacija: [1.64226088 1.87796998]
Registar: 3 difuzija i relaksacija: [1.67204264 1.94214045]
Registar: 4 difuzija i relaksacija: [1.65125191 1.97229491]
Registar: 5 difuzija i relaksacija: [1.65297383 1.94540403]
Registar: 6 difuzija i relaksacija: [1.62262476 1.86586342]
Registar: 7 difuzija i relaksacija: [0.92885001 2.09568088]
Log legalne mase: -4.8561365286113345

loto7_4716_k89_loto_2980.csv redova: 2979
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.001 validacioni NLL: 17.646604218804516
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.01 validacioni NLL: 17.68792118763025
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Test kola: 447
Test NLL: 17.61833404236582
Test srednji broj pogodaka: 1.2460850111856823
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Registar: 1 difuzija i relaksacija: [0.92190777 1.9455899 ]
Registar: 2 difuzija i relaksacija: [1.6163954  1.82444815]
Registar: 3 difuzija i relaksacija: [1.65989864 1.94416863]
Registar: 4 difuzija i relaksacija: [1.65302738 1.96826857]
Registar: 5 difuzija i relaksacija: [1.65282543 1.9351863 ]
Registar: 6 difuzija i relaksacija: [1.61848273 1.89638191]
Registar: 7 difuzija i relaksacija: [0.89174458 2.05406182]
Log legalne mase: -4.999025200567035

loto7_4716_k89_loto_plus_1736.csv redova: 1735
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.001 validacioni NLL: 17.65889913619248
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.01 validacioni NLL: 17.69338068778296
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Test kola: 261
Test NLL: 17.541747450487968
Test srednji broj pogodaka: 1.3026819923371646
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Registar: 1 difuzija i relaksacija: [0.88010615 1.95146082]
Registar: 2 difuzija i relaksacija: [1.65255951 1.81425667]
Registar: 3 difuzija i relaksacija: [1.64709344 1.93097933]
Registar: 4 difuzija i relaksacija: [1.62555984 1.89455281]
Registar: 5 difuzija i relaksacija: [1.64491277 1.97351327]
Registar: 6 difuzija i relaksacija: [1.55837843 1.77914027]
Registar: 7 difuzija i relaksacija: [0.90688144 1.99198109]
Log legalne mase: -5.058816782685961


loto7_4716_k89.csv
NEXT: 1 x 17 y 29 z 39

loto7_4716_k89_loto_2980.csv
NEXT: 1 x 17 y 26 z 39

loto7_4716_k89_loto_plus_1736.csv
NEXT: 2 x 17 y 24 z 39
"""



"""
1. 
Ising Model	
Spinovi, međusobne interakcije, Hamiltonijan i temperaturna dinamika. 
Koristim tačan zbir nad 64 spinska stanja, uz očuvanje Hamiltonijana i temperaturne raspodele.

2. 
ODE - Ordinary Differential Equation	
Evolucija stanja preko zadate diferencijalne jednačine i početnog uslova, uz solve_ivp.	
Diferencijalna dinamika. Desnu stranu jednačine naučiti iz prethodnih kola i proveriti evoluciju raspodele. 
Otvoreni kvantni model, jer evolucija raspodele može biti neunitarna. 

3. 
Partial Differential Equations	
PDE — Heat/Diffusion
Difuzija kroz jednačinu toplote, rešavana konačnim razlikama i vremenskim koracima. 
Za Loto treba sačuvati difuzioni operator, vremensku evoluciju, početne i granične uslove, uz učenje parametara iz istorije. 
Posebno treba proveriti očuvanje ukupne verovatnoće. 
Otvoreni kvantni model, jer evolucija raspodele može biti neunitarna. 

4. 
Fourier Transform	
DFT/FFT, kompleksni koeficijenti, spektralne amplitude, faze i inverzna transformacija.	
Poseban spektralni model koji uči vremensku evoluciju komponenti raspodele. 
Fourier uči spektralne amplitude i faze. 
U Fourier kodu FFT računanje filtera, jer množenje gustih Fourier matrica može ostaviti grešku poništavanja i kod identičnog filtera. 

5, 
Quantum Mechanics	
Kompleksna talasna funkcija, Schrödingerova evolucija, potencijal, interferencija i Bornove verovatnoće.	
Kvantni dinamički model sa početnim amplitudama i parametrima naučenim iz istorije. 
Schrödingerov model uci Hermitski Hamiltonijan.



Uključene optimizacija i vizuelizacija u svaki odgovarajući model. 

Plotting and Data Visualization	
Prikaz podataka, histogrami, dijagrami i toplotne mape.	
Dijagnostika unutar prediktivnih fajlova. 
Vizuelizacija ulazi u dijagnostiku — prikaz naučene raspodele, promene kroz vreme, konvergencije i rezultata odvojenog testa.

Optimization	
Optimization ulazi kao stvarni postupak obuke. 
Gradijentni spust, promenljiv korak, konjugovane gradijente i Nelder-Mead. 
Te postupke odabrati prema ciljnoj funkciji i hronološkoj validaciji. 
Poređenje više determinističkih početnih tačaka čuva istraživanje različitih lokalnih minimuma. 
Takvo poređenje samo po sebi ne potvrđuje globalni optimum.





Fajl	                Osnovna ideja 	                                                                                Šta treba učiti iz istorije
Ising	                Spinovi, Hamiltonijan, međusobne interakcije i temperaturna raspodela.	                        Jačine interakcija, lokalna polja i njihova zavisnost od prethodnih kola.
ODE	                    Početno stanje, diferencijalna jednačina i integracija vremenske evolucije.	                    Parametri jednačine koja opisuje prelaz raspodele u sledeće kolo.
PDE — Heat/Diffusion	Difuzioni operator, konačne razlike, vremenski koraci i granični uslovi.	                    Parametri difuzije i eventualni izvorni član, uz očuvanje nenegativnosti i ukupne verovatnoće.
Fourier	                Kompleksni spektralni koeficijenti, amplitude, faze i inverzna transformacija.	                Evolucija spektralnih komponenti kroz kola.
Quantum Mechanics	    Schrödingerova evolucija, kompleksne amplitude, Hamiltonijan, interferencija i Bornovo pravilo.	Početne amplitude i parametri Hamiltonijana koji opisuju istorijske prelaze.


Za kvantnu realizaciju posebno treba očuvati razliku između difuzije i Schrödingerove evolucije. Difuzija je neunitarna. 
Njeno preslikavanje na Qiskit zahteva odgovarajuću konstrukciju. Sama zamena difuzionog operatora unitarnim kolom promenila bi metod.

Svih pet fajlova obrađuje ista tri CSV-a zasebno, sa jednom NEXT kombinacijom po CSV-u. Razlikovaće se po naučenom modelu i postupku evolucije. 



Svaki model koristi prethodno kolo kao početno stanje. 
Obuka obuhvata sve raspoložive istorijske prelaze.

- Ising koristi tačan zbir nad 64 spinska stanja.
- ODE koristi naučeni generator prenosa verovatnoće.
- PDE koristi konačne razlike, periodične granice i izvorni član koji čuva ukupnu verovatnoću.
- Fourier uči spektralne amplitude i faze. Njegov kvantni filter koristi jedan pomoćni kubit.
- Schrödingerov model uči Hermitski Hamiltonijan i čuva unitarnu evoluciju.


Optimizacija koristi konjugovane gradijente. 
Svaki fajl ima hronološku validaciju, odvojeni test, završnu obuku i grafikone.



Statička i matematička provera: 

Kod	            Nalaz

Ising	        Logaritam računati iz energije i particione funkcije. 
                Znak Hamiltonijana, Gibbsova normalizacija, gradijenti polja i interakcija, logaritamski izlaz.

ODE	            Izveo sam gradijent generatora i proverio konstrukciju prenosa. 
                Smer prenosa, zbir kolona generatora, Fréchet gradijent i konstrukcija skočnih operatora.

PDE	            Proverio sam FTCS korak, gradijent izvora i kvantni kanal. 
                Stabilnost FTCS koraka, očuvanje verovatnoće, gradijent izvora i kvantni kanal.

Fourier	        Filter  računa FFT. Time se izbegava poništavanje pri množenju gustih Fourier matrica. Logaritmi se računaju iz amplituda.
                Smer FFT/QFT, normalizacija filtera, gradijenti amplituda i faza, raspored pomoćnog kubita. 

Schrödinger     Evolucija koristi direktni expm, a gradijent odgovarajući Fréchet izvod. Logaritmi se računaju iz amplituda.
                Hermitičnost, matrična eksponencijala, kompleksni adjoint gradijent i logaritmi amplituda.


Provereni su i svi pozivi joint() — Ising, Fourier i Schrödinger verzije prosleđuju logaritme. ODE i PDE prosleđuju verovatnoće svojoj odgovarajućoj funkciji.
Hronološke granice treninga, validacije i testa pravilno su indeksirane. Završna obuka obuhvata sve istorijske prelaze.
Konvergencija optimizatora i prolazak ugrađenih provera.
"""
