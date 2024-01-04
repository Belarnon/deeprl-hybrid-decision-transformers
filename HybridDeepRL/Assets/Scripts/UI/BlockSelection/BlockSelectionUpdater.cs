using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PKB.App;

namespace PKB.UI
{

    /// <summary>
    /// A <see cref="BlockSelectionUpdater"/> is a component that updates the block selection UI when the game state changes.
    /// </summary>
    [RequireComponent(typeof(BlockSelectionPanel))]
    [AddComponentMenu("PKB/UI/Block Selection Updater")]
    public class BlockSelectionUpdater : MonoBehaviour
    {
        #region Inspector Interface

        /// <summary>
        /// The game manager that holds the game state.
        /// </summary>
        [Tooltip("The game manager that holds the game state.")]
        [SerializeField]
        private GameManager gameManager;

        #endregion

        #region Internal State

        private BlockSelectionPanel blockSelectionPanel;

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            blockSelectionPanel = GetComponent<BlockSelectionPanel>();
            gameManager.onGameStep.AddListener(UpdateBlocks);
            gameManager.onGameReset.AddListener(UpdateBlocks);
        }

        #endregion

        #region Public Interface

        public void UpdateBlocks()
        {
            blockSelectionPanel.SetBlocks(gameManager.GetAvailableBlocks());
        }

        #endregion
    }
}
