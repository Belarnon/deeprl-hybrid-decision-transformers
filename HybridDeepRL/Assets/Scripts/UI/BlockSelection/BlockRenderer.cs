using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Assertions;
using GameLogic;

namespace PKB.UI
{

    /// <summary>
    /// A <see cref="BlockRenderer"/> is a component that renders a given block/tile in the UI.
    /// </summary>
    [RequireComponent(typeof(RectTransform))]
    [AddComponentMenu("PKB/UI/Block Renderer")]
    public class BlockRenderer : MonoBehaviour
    {
        #region Inspector Interface

        /// <summary>
        /// Describes how to render the blocks/tiles.
        /// </summary>
        [Tooltip("Describes how to render the blocks/tiles.")]
        [SerializeField]
        private TileStyle tileStyle;

        /// <summary>
        /// If set, this block will be rendered by default (optional).
        /// </summary>
        [Tooltip("If set, this block will be rendered by default (optional).")]
        public BlockScriptableObject defaultBlock;

        /// <summary>
        /// The parent <see cref="RectTransform"/> to use for the tiles.
        /// </summary>
        /// <remarks>
        /// This will be dragged around to move the entire tile.
        /// </remarks>
        [Tooltip("The parent RectTransform to use for the tiles.")]
        [SerializeField]
        private RectTransform tileParent;

        

        #endregion

        #region Internal State

        private BlockScriptableObject currentBlock;

        private int currentBlockIndex;
        
        private bool blockDirty = false;

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            CheckDependencies();
            if (defaultBlock != null)
            {
                SetBlock(defaultBlock);
            }
        }

        private void Update()
        {
            if (blockDirty)
            {
                UpdateBlock();
                blockDirty = false;
            }
        }

        #endregion

        #region Public Interface

        /// <summary>
        /// Sets the block to render.
        /// </summary>
        /// <param name="block">The block to render.</param>
        public void SetBlock(BlockScriptableObject block, int index = -1)
        {
            currentBlock = block;
            currentBlockIndex = index;
            blockDirty = true;
        }

        public void ClearBlock()
        {
            currentBlock = null;
            currentBlockIndex = -1;
            blockDirty = true;
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Updates the block UI.
        /// </summary>
        private void UpdateBlock()
        {
            RemoveBlock();
            if (currentBlock != null)
            {
                DrawBlock();
            }
        }

        private void RemoveBlock()
        {
            foreach (Transform child in tileParent.transform)
            {
                Destroy(child.gameObject);
            }
        }

        private void DrawBlock()
        {
            Vector2Int center = currentBlock.getCenter();
            Vector2 centerOffset = ComputeCenterOffset(center);
            foreach (Vector2Int offset in currentBlock.getBlockOffsets())
            {
                DrawCell(center + offset, offset, centerOffset);
            }
        }

        private void DrawCell(Vector2Int coordinates, Vector2Int coordinateCenterOffset, Vector2 offset = default)
        {
            // Create a new child with an image component.
            GameObject cell = new($"Cell ({coordinates.x}, {coordinates.y})");
            cell.transform.SetParent(tileParent.transform);
            Image image = cell.AddComponent<Image>();

            // Set the tile image and color.
            image.sprite = tileStyle.cellSprite;
            image.color = tileStyle.cellColor;

            // Set the size and position of the tile.
            RectTransform rectTransform = cell.GetComponent<RectTransform>();
            rectTransform.sizeDelta = tileStyle.cellSize;
            rectTransform.anchoredPosition = new Vector2(
                coordinates.x * (tileStyle.cellSize.x + tileStyle.cellOffset.x),
                coordinates.y * (tileStyle.cellSize.y + tileStyle.cellOffset.y)
            ) + offset;

            // Add a tile cell component and configure it.
            TileCell tileCell = cell.AddComponent<TileCell>();
            tileCell.SetCenterOffset(coordinateCenterOffset);
            tileCell.SetBlockIndex(currentBlockIndex);
        }

        private Vector2 ComputeCenterOffset(Vector2Int centerCoordinates)
        {
            return new Vector2(
                -centerCoordinates.x * (tileStyle.cellSize.x + tileStyle.cellOffset.x),
                -centerCoordinates.y * (tileStyle.cellSize.y + tileStyle.cellOffset.y)
            );
        }

        private void CheckDependencies()
        {
            // Make sure the cell prefab is set.
            Assert.IsNotNull(tileStyle, "Tile style must be set.");
            Assert.IsNotNull(tileParent, "Tile parent must be set.");
        }

        #endregion

    }
}
