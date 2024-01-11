using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace PKB.Recording
{

    /// <summary>
    /// The <see cref="Trajectory"/> class represents a sequence of <see cref="Transition"/>s.
    /// </summary>
    [System.Serializable]
    public class Trajectory
    {
        /// <summary>
        /// The transitions of the agent.
        /// </summary>
        public List<Transition> transitions = new();
    }
}
