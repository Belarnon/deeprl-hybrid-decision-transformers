using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace PKB.Recording
{

    /// <summary>
    /// The <see cref="DemonstrationRecord"/> class represents a sequence of <see cref="Trajectory"/>s.
    /// </summary>
    [System.Serializable]
    public class DemonstrationRecord
    {
        /// <summary>
        /// The trajectories of the agent.
        /// </summary>
        public List<Trajectory> trajectories = new();
    }
}
