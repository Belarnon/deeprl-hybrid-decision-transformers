using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

namespace PKB.Recording
{
    public abstract class RecordableAgent : Agent
    {
        #region Inspector Interface

        /// <summary>
        /// Fired when the agent's trajectory is started.
        /// </summary>
        [Tooltip("Fired when the agent's trajectory is started.")]
        public UnityEvent OnTrajectoryStarted;

        /// <summary>
        /// Fired when the agent's trajectory is ended.
        /// </summary>
        [Tooltip("Fired when the agent's trajectory is ended.")]
        public UnityEvent OnTrajectoryEnded;

        /// <summary>
        /// Fired when the agent's observation is collected.
        /// </summary>
        [Tooltip("Fired when the agent's observation is collected.")]
        public UnityEvent<VectorSensor> OnObservationCollected;
        
        /// <summary>
        /// Fired when the agent's action is selected.
        /// </summary>
        [Tooltip("Fired when the agent's action is selected.")]
        public UnityEvent<ActionBuffers> OnActionSelected;

        /// <summary>
        /// Fired when the agent's reward is received.
        /// </summary>
        [Tooltip("Fired when the agent's reward is received.")]
        public UnityEvent<float> OnRewardReceived;

        #endregion

        #region Internal State

        private bool hasUnfinishedTrajectory = false;
        private bool finalObservationCollected = false;

        #endregion

        #region Unity Lifecycle

        protected override void OnDisable()
        {
            base.OnDisable();
            if (hasUnfinishedTrajectory)
            {
                MarkTrajectoryEnd();
            }
        }

        protected virtual void OnApplicationQuit()
        {
            if (hasUnfinishedTrajectory)
            {
                MarkTrajectoryEnd();
            }
        }

        #endregion

        #region Protected Interface

        /// <summary>
        /// Marks the start of a trajectory.
        /// </summary>
        protected void MarkTrajectoryStart()
        {
            if (hasUnfinishedTrajectory)
            {
                Debug.LogWarning("Marking the start of a trajectory while another one is still unfinished. Marking the end of the previous trajectory.");
                MarkTrajectoryEnd();
            }
            SetTrajectoryBeginState();
            OnTrajectoryStarted?.Invoke();
        }

        /// <summary>
        /// Marks the end of a trajectory.
        /// </summary>
        protected void MarkTrajectoryEnd()
        {
            if (!hasUnfinishedTrajectory)
            {
                Debug.LogWarning("Marking the end of a trajectory while none has been started. Doing nothing.");
                return;
            }
            hasUnfinishedTrajectory = false;
            OnTrajectoryEnded?.Invoke();
        }

        /// <summary>
        /// Commits the observation for other components to use.
        /// </summary>
        /// <param name="sensor">The sensor to commit.</param>
        protected void CommitObservation(VectorSensor sensor)
        {
            if (!CanCommitObservation())
            {
                return;
            }
            OnObservationCollected?.Invoke(sensor);
        }

        /// <summary>
        /// Commits the action for other components to use.
        /// </summary>
        /// <param name="actionBuffers">The action buffers to commit.</param>
        protected void CommitAction(ActionBuffers actionBuffers)
        {
            if (!CanCommitAction())
            {
                return;
            }
            OnActionSelected?.Invoke(actionBuffers);
        }

        /// <summary>
        /// Commits the reward for other components to use.
        /// </summary>
        /// <param name="reward">The reward to commit.</param>
        protected void CommitReward(float reward)
        {
            if (!CanCommitReward())
            {
                return;
            }
            OnRewardReceived?.Invoke(reward);
        }

        #endregion

        #region Helpers

        private void SetTrajectoryBeginState()
        {
            hasUnfinishedTrajectory = true;
            finalObservationCollected = false;
        }

        private bool CanCommitObservation()
        {
            if (hasUnfinishedTrajectory)
            {
                return true;
            }
            if (finalObservationCollected)
            {
                Debug.LogError("Committing an observation while no trajectory is started or the final observation has already been collected. Doing nothing.");
                return false;
            }
            else
            {
                finalObservationCollected = true;
                return true;
            }
        }

        private bool CanCommitAction()
        {
            if (!hasUnfinishedTrajectory)
            
            {
                Debug.LogError("Committing an action while no trajectory is started.");
                return false;
            }
            return true;
        }

        private bool CanCommitReward()
        {
            if (!hasUnfinishedTrajectory)
            {
                Debug.LogError("Committing a reward while no trajectory is started.");
                return false;
            }
            return true;
        }


        #endregion
    }
}
