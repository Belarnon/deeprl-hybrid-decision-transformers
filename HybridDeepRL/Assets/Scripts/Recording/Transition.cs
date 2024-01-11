using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace PKB.Recording
{

    /// <summary>
    /// The <see cref="Transition"/> class represents a single state, action, reward tuple.
    /// </summary>
    [System.Serializable]
    public class Transition
    {

        /// <summary>
        /// The observation vector of the agent.
        /// </summary>
        public float[] observation;

        /// <summary>
        /// The action vectors of the agent.
        /// </summary>
        public Action action;

        /// <summary>
        /// The reward received by the agent.
        /// </summary>
        public float reward;
        
    }
}
