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

        #endregion

        #region Internal State

        private BlockScriptableObject currentBlock;

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

        #endregion

        #region Public Interface

        /// <summary>
        /// Sets the block to render.
        /// </summary>
        /// <param name="block">The block to render.</param>
        public void SetBlock(BlockScriptableObject block)
        {
            currentBlock = block;
            UpdateBlock();
        }

        public void ClearBlock()
        {
            currentBlock = null;
            UpdateBlock();
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
            foreach (Transform child in transform)
            {
                Destroy(child.gameObject);
            }
        }

        private void DrawBlock()
        {
            Vector2Int center = currentBlock.getCenter();
            Vector2 centerOffset = ComputeCenterOffset(center);
            DrawCell(center, centerOffset);
            foreach (Vector2Int offset in currentBlock.getBlockOffsets())
            {
                DrawCell(center + offset, centerOffset);
            }
        }

        private void DrawCell(Vector2Int coordinates, Vector2 offset = default)
        {
            // Create a new child with an image component.
            GameObject cell = new GameObject($"Cell ({coordinates.x}, {coordinates.y})");
            cell.transform.SetParent(transform);
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
        }

        #endregion

    }
}
