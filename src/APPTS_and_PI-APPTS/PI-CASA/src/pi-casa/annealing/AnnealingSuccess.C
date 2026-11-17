// Copyright 2008, 2009 Brady J. Garvin

// This file is part of Covering Arrays by Simulated Annealing (CASA).

// CASA is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// CASA is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with CASA.  If not, see <http://www.gnu.org/licenses/>.


#include <ctime>
#include <iostream>
#include "io/Usage.H"
#include "io/OutputFile.H"
#include "covering/space/CoveringArraySpace.H"
#include "annealing/AnnealingSuccess.H"

using namespace std;

bool AnnealingSuccess::operator ()(unsigned rows) const {
  if (double(clock() - start) / CLOCKS_PER_SEC >= double(runtime)) {
    return false;
  }
  const clock_t begun = clock();
  const CoveringArraySpace*space =
    dynamic_cast<const CoveringArraySpace*>(search.getSpace());
  assert(space);
  cout << "Trying " << rows << " rows" << endl;
  cout << "Building start state" << endl;
  CoveringArray startState = space->createStartState(rows);
  unsigned lastNoncoverage = startState.getNoncoverage().size();
  const unsigned startNoncoverage = lastNoncoverage;
  const clock_t built = clock();
  search.addStartState(startState);
  cout << "Searching" << endl;
  set<NodeT*>solutions = search.search(iterations, false);
  const clock_t returned = clock();
  set<const NodeT*>best = search.getBest();
  if (!best.empty()) {
    lastNoncoverage = (*best.begin())->getState().getNoncoverage().size();
  }
  const bool complete = !solutions.empty();
  const double foundCpu = double(returned - start) / CLOCKS_PER_SEC;
  const bool accepted = complete && foundCpu <= double(runtime);
  clock_t writeCost = 0;
  if (accepted) {
    cout << "Met coverage with " << rows << " rows" << endl;
    if (!solution.getRows() || rows < solution.getRows()) {
      solution = (*solutions.begin())->getState();
      costTime = foundCpu;
      OutputFile output(outputFile ? outputFile : "anneal.out");
      output.setCoveringArray(solution);
      const clock_t writing = clock();
      output.write();
      writeCost = clock() - writing;
      cerr << "[PI-CASA] event=incumbent size=" << rows
           << " best_found_cpu_s=" << costTime
           << " time_event=native_search_return"
           << endl;
    }
  } else {
    cout << "Failed to meet coverage with " << rows << " rows" << endl;
  }
  cerr << "[PI-CASA] event=attempt n_pred=" << predictedSize
       << " target=" << rows << " complete=" << complete
       << " accepted=" << accepted << " incumbent=" << solution.getRows()
       << " start_noncoverage=" << startNoncoverage
       << " best_noncoverage=" << lastNoncoverage
       << " iteration_limit=" << iterations
       << " build_cpu_s=" << double(built - begun) / CLOCKS_PER_SEC
       << " search_and_feedback_cpu_s=" << double(returned - built) / CLOCKS_PER_SEC
       << " write_cpu_s=" << double(writeCost) / CLOCKS_PER_SEC
       << " process_cpu_s=" << double(clock() - start) / CLOCKS_PER_SEC
       << endl;
  search.clear();
  return accepted;
}
