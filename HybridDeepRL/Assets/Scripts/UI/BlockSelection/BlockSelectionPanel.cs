using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using GameLogic;

namespace PKB.UI
{
    /// <summary>
    /// A <see cref="BlockSelectionPanel"/> is a panel that displays the sampled blocks from the game 
    /// </summary>
    [AddComponentMenu("PKB/UI/Block Selection Panel")]
    [RequireComponent(typeof(RectTransform))]
    public class BlockSelectionPanel : MonoBehaviour
    {
        
        #region Inspector Interface

        /// <summary>
        /// The block panel prefab to use to display each block.
        /// </summary>
        [Tooltip("The block panel prefab to use to display each block.")]
        [SerializeField]
        private GameObject blockPanelPrefab;

        /// <summary>
        /// The maximum number of blocks that can be selected for placement.
        /// </summary>
        [Tooltip("The maximum number of blocks that can be selected for placement.")]
        [Range(1, 5)]
        [SerializeField]
        private int numberOfBlocksToSelect = 3;

        #endregion

        #region Internal State

        private List<GameObject> blockPanels = new();

        private List<BlockRenderer> blockRenderers = new();

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            CheckDependencies();
            ResetBlockPanels();
        }

        #endregion

        #region Public Interface

        /// <summary>
        /// Changes the number of individual blocks/tiles that can be selected.
        /// </summary>
        /// <param name="numberOfBlocksToSelect">The number of blocks that can be selected.</param>
        public void SetNumberOfBlocksToSelect(int numberOfBlocksToSelect)
        {
            this.numberOfBlocksToSelect = numberOfBlocksToSelect;
            ResetBlockPanels();
        }

        public void SetBlocks(BlockScriptableObject[] blocks)
        {
            Assert.IsTrue(blocks.Length <= numberOfBlocksToSelect, 
                $"Cannot set {blocks.Length} blocks in a panel that can only display {numberOfBlocksToSelect} blocks.");
            for (int i = 0; i < blocks.Length; i++)
            {
                BlockScriptableObject block = blocks[i];
                if (block != null)
                {
                    blockRenderers[i].SetBlock(block);
                }
                else
                {
                    blockRenderers[i].ClearBlock();
                }
            }
        }

        #endregion

        #region Helpers

        private void ResetBlockPanels()
        {
            ClearBlockPanels();
            CreateBlockPanels();
        }

        private void ClearBlockPanels()
        {
            foreach (GameObject blockPanel in blockPanels)
            {
                Destroy(blockPanel);
            }
        }

        private void CreateBlockPanels()
        {
            blockPanels = new List<GameObject>();
            blockRenderers = new List<BlockRenderer>();
            for (int i = 0; i < numberOfBlocksToSelect; i++)
            {
                GameObject blockPanel = Instantiate(blockPanelPrefab, transform);
                blockPanels.Add(blockPanel);
                BlockRenderer blockRenderer = blockPanel.GetComponent<BlockRenderer>();
                Assert.IsNotNull(blockRenderer, $"Block panel {blockPanel.name} has no block renderer.");
                blockRenderers.Add(blockRenderer);
            }
        }

        private void CheckDependencies()
        {
            Assert.IsNotNull(blockPanelPrefab, $"Block selection panel {name} has no block panel prefab.");
        }

        #endregion

    }
}
