"""Independent CSV/backtest oracle and executable-level integration tests (stdlib only)."""
import csv
import math
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist


def rows(path):
    with open(path, newline='') as file:
        result = list(csv.DictReader(file))
    assert all(None not in row for row in result), f'malformed CSV: {path}'
    return result


def run(exe, dataset, out, *args):
    subprocess.run([str(exe), '--data', str(dataset), '--ticker', 'TEST',
                    '--window-days', '5', '--out', str(out), *args], check=True, capture_output=True, text=True)
    return rows(out / 'forecasts.csv')


def fixture(folder):
    folder.mkdir()
    (folder / 'assets.csv').write_text('ticker\nTEST\n')
    rng = random.Random(921)
    history = {}
    with open(folder / 'prices_15m.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
        for day in range(9):
            price = 100.0
            for slot in range(28):
                time = datetime(2026, 10, 5, 10, 15, tzinfo=timezone.utc) + timedelta(days=day, minutes=15*slot)
                r = rng.gauss(0, .002 * (1 + day / 9))
                close = price * math.exp(r)
                writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), 'TEST', price,
                                 max(price, close), min(price, close), close, 1000])
                history[int(time.timestamp())] = r
                price = close
    return history


def verify_output(out, exposure, confidence):
    forecasts = rows(out / 'forecasts.csv')
    groups = defaultdict(list)
    for row in forecasts:
        groups[row['model']].append(row)
        q, target = float(row['quantile']), float(row['target_return'])
        assert math.isclose(float(row['var']), -exposure * math.expm1(q), rel_tol=1e-12)
        assert math.isclose(float(row['realized_pnl']), exposure * math.expm1(target), rel_tol=1e-12, abs_tol=1e-12)
    assert set(groups) == {'Realized-CAViaR', 'EWMA-N', 'GARCH-t'}
    target_sets = [{r['target_time'] for r in group if r['evaluated'] == '1'} for group in groups.values()]
    assert target_sets[0] == target_sets[1] == target_sets[2]
    assert len(target_sets[0]) == 4*24
    for summary in rows(out / 'backtest_summary.csv'):
        group = [r for r in groups[summary['model']] if r['evaluated'] == '1']
        sign = -1 if exposure < 0 else 1
        errors = [sign * (float(r['target_return']) - float(r['quantile'])) for r in group]
        hits = sum(error < 0 for error in errors)
        assert hits == int(summary['exceptions'])
        assert int(summary['n']) == len(group)
        expected = sum(error * (1-confidence - (error < 0)) for error in errors) / len(group)
        assert math.isclose(float(summary['mean_pinball']), expected, rel_tol=1e-12)
        assert math.isclose(float(summary['rate']), hits/len(group), abs_tol=1e-15)
    for row in rows(out / 'benchmark.csv'):
        assert 0 <= float(row['p50_ns']) <= float(row['p95_ns']) <= float(row['p99_ns'])
        assert int(row['state_bytes']) > 0
    assert len(rows(out / 'risk_15m.csv')) == len(forecasts)
    return forecasts


def verify_fit(out, history, confidence):
    """Reconstruct features and recursive training objective independently of C++."""
    features = []
    times = sorted(history)
    for i, time in enumerate(times):
        if i % 28 < 3 or i % 28 == 27:
            continue
        past = [history[t] for t in times[i-3:i+1]]
        features.append((time, time+900, sum(past), sum(r*r for r in past), history[time+900]))
    for row in rows(out / 'params_daily.csv'):
        if row['model'] != 'Realized-CAViaR':
            continue
        start, end, effective = map(int, (row['training_start'], row['training_end'], row['effective_time']))
        train = [f for f in features if start <= f[0] < effective]
        targets = sorted(f[4] for f in train if f[1] <= end)
        tau = 1-confidence
        initial = targets[math.floor(tau * (len(targets)-1))]
        assert math.isclose(float(row['initial_quantile']), initial, abs_tol=1e-14)
        beta = [float(row[f'beta{i}']) for i in range(4)]
        q, loss, constant = initial, 0., 0.
        for _, target_time, aggregate, rv, target in train:
            q = beta[0] + beta[1]*q + beta[2]*abs(aggregate) + beta[3]*math.sqrt(rv)
            if target_time <= end:
                error = target-q
                loss += error * (tau - (error < 0))
                error = target-initial
                constant += error * (tau - (error < 0))
        loss /= len(targets)
        constant /= len(targets)
        assert math.isclose(float(row['mean_pinball']), loss, abs_tol=1e-13)
        assert math.isclose(float(row['last_quantile']), q, abs_tol=1e-13)
        assert math.isclose(float(row['constant_pinball']), constant, abs_tol=1e-13)
        assert loss <= constant + 1e-13


def high_frequency(exe, root):
    rng = random.Random(17)
    prices, price = [], 100.
    for i in range(20001):
        if i:
            price *= math.exp(rng.gauss(0, .00015))
        prices.append((200+i*5, price))  # nonzero minute phase, crossing midnight
    path = root / 'five_seconds.csv'
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'price'])
        writer.writerows(prices)
    out = root / 'hf'
    subprocess.run([str(exe), '--prices', str(path), '--out', str(out)], check=True, capture_output=True)
    forecasts = rows(out / 'forecasts.csv')
    assert {r['model'] for r in forecasts} == {'Realized-CAViaR', 'EWMA-N', 'GARCH-t'}
    parameters = rows(out / 'params_daily.csv')
    assert len(parameters) == 11  # CAViaR + 5 EWMA + 5 GARCH phases
    variances = {int(r['phase']): float(r['ewma_variance']) for r in parameters if r['model'] == 'EWMA-N'}
    cutoff = prices[len(prices)//2][0]
    tail = NormalDist().inv_cdf(1-.99)
    selected = []
    for row in forecasts:
        time = int(row['forecast_time'])
        i = (time-200)//5
        assert int(row['target_time']) == time+300
        if row['model'] == 'Realized-CAViaR':
            expected = math.log(prices[i][1])-math.log(prices[i-60][1])
            assert math.isclose(float(row['aggregate_return']), expected, abs_tol=1e-13)
            if i+60 < len(prices):
                expected = math.log(prices[i+60][1])-math.log(prices[i][1])
                assert math.isclose(float(row['target_return']), expected, abs_tol=1e-13)
            else:
                assert math.isnan(float(row['target_return']))
                assert row['exception'] == '' and row['evaluated'] == '0'
            if row['evaluated'] == '1':
                selected.append((time, time+300))
        if row['model'] == 'EWMA-N':
            phase = (time-200)//60 % 5
            if time > cutoff:
                r = float(row['aggregate_return'])
                variances[phase] = .94*variances[phase]+.06*r*r
            expected = tail * math.sqrt(variances[phase])
            assert math.isclose(float(row['quantile']), expected, rel_tol=1e-12)
    assert all(a[1] <= b[0] for a, b in zip(selected, selected[1:]))
    assert len(rows(out / 'risk_5m.csv')) == len(forecasts)
    for model in ('Realized-CAViaR', 'EWMA-N', 'GARCH-t'):
        assert [(r['forecast_time'], r['evaluated']) for r in forecasts if r['model'] == model] == [
            (r['forecast_time'], r['evaluated']) for r in forecasts if r['model'] == 'Realized-CAViaR']
    # Explicit session gap resets features and prevents future labels crossing it.
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'price'])
        writer.writerows((t+(86400 if i>=10000 else 0), p) for i, (t, p) in enumerate(prices))
    rejected = subprocess.run([str(exe), '--prices', str(path), '--out', str(root/'rejected')], capture_output=True)
    assert rejected.returncode != 0
    subprocess.run([str(exe), '--prices', str(path), '--reset-on-gap', '--refit', 'daily', '--out', str(root/'reset')], check=True, capture_output=True)


def main():
    exe = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix='rcaviar-contract-') as directory:
        root = Path(directory)
        data = root / 'data'
        history = fixture(data)
        run(exe, data, root / 'daily', '--refit', 'daily')
        daily = verify_output(root / 'daily', 100000, .99)
        verify_fit(root / 'daily', history, .99)
        params = rows(root / 'daily' / 'params_daily.csv')
        calendars = defaultdict(list)
        for row in params:
            assert int(row['training_end']) < int(row['effective_time'])
            calendars[row['model']].append((row['effective_time'], row['training_start'], row['training_end']))
        assert len(calendars['Realized-CAViaR']) == 4
        assert calendars['Realized-CAViaR'] == calendars['EWMA-N'] == calendars['GARCH-t']
        assert len({r[1] for r in calendars['Realized-CAViaR']}) == 4
        run(exe, data, root / 'once', '--refit', 'once')
        assert len(rows(root / 'once' / 'params_daily.csv')) == 3
        run(exe, data, root / 'short', '--refit', 'once', '--exposure', '-200000', '--confidence', '.95')
        verify_output(root / 'short', -200000, .95)
        # Changing strictly future data cannot change the first prediction or calibration.
        first_time = int(daily[0]['forecast_time'])
        price_rows = rows(data / 'prices_15m.csv')
        for row in price_rows:
            time = int(datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp())
            if time > first_time:
                row['close'] = str(float(row['close'])*1.0001)
        with open(data / 'prices_15m.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=price_rows[0].keys())
            writer.writeheader()
            writer.writerows(price_rows)
        changed = run(exe, data, root / 'changed', '--refit', 'daily')
        for before, after in zip(daily[:3], changed[:3]):
            assert before['model'] == after['model']
            assert before['quantile'] == after['quantile']
        for args in [('--confidence', '1'), ('--lambda', '0'), ('--refit', 'weekly'),
                     ('--exposure', 'nan'), ('--window-days', '4x'), ('--unknown', '1')]:
            proc = subprocess.run([str(exe), *args], capture_output=True)
            assert proc.returncode != 0, args
        high_frequency(exe, root)
    print('CSV contract, independent pinball/fit oracle, daily rolling schedule, shorts and causality passed')


if __name__ == '__main__':
    main()
