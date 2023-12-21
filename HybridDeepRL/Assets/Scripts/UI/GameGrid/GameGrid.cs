using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Assertions;

namespace PKB.UI
{
    public class GameGrid : MonoBehaviour
    {
        #region Inspector Interface

        /// <summary>
        /// The UI prefab to use for a grid cell.
        /// </summary>
        [Tooltip("The UI prefab to use for a grid cell.")]
        [SerializeField]
        private GameObject gridCell;

        /// <summary>
        /// The color to use for empty cells.
        /// </summary>
        [Tooltip("The color to use for empty cells.")]
        [SerializeField]
        private Color emptyColor = new Color(0.25f, 0.25f, 0.25f, 1.0f);

        /// <summary>
        /// The color to use for cells that are filled.
        /// </summary>
        [Tooltip("The color to use for cells that are filled.")]
        [SerializeField]
        private Color filledColor = new Color(0.5f, 0.5f, 0.5f, 1.0f);

        /// <summary>
        /// The size in pixels of a grid cell.
        /// </summary>
        [Tooltip("The size in pixels of a grid cell.")]
        [SerializeField]
        private Vector2 cellSize = new Vector2(64, 64);

        /// <summary>
        /// The spacing in pixels between grid cells.
        /// </summary>
        [Tooltip("The spacing in pixels between grid cells.")]
        [SerializeField]
        private Vector2 cellSpacing = new Vector2(4, 4);

        /// <summary>
        /// Whether or not to recenter the grid when the grid is created or resized.
        /// </summary>
        [Tooltip("Whether or not to recenter the grid when the grid is created or resized.")]
        [SerializeField]
        private bool recenterGrid = true;

        #endregion

        #region Internal State

        /// <summary>
        /// The grid cells.
        /// </summary>
        private GameObject[,] cells;

        /// <summary>
        /// The size of the grid.
        /// </summary>
        private Vector2Int gridSize;

        /// <summary>
        /// The initial position of the grid parent.
        /// </summary>
        private Vector2 initialPosition;

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            // Make sure the cell prefab is set.
            Assert.IsNotNull(gridCell, "The grid cell prefab must be set.");
            initialPosition = transform.localPosition;
            CreateGrid(new Vector2Int(10, 10));
        }

        #endregion

        #region Public Interface

        /// <summary>
        /// Creates a new grid with the given size.
        /// </summary>
        /// <param name="gridSize">The size of the grid.</param>
        public void CreateGrid(Vector2Int gridSize)
        {
            Assert.IsTrue(gridSize.x > 0 && gridSize.y > 0, "The grid size must be greater than 0 in both dimensions.");
            this.gridSize = gridSize;
            DestroyGrid();
            CreateEmptyGrid(gridSize);
            if (recenterGrid)
            {
                CenterGrid();
            }
        }

        /// <summary>
        /// Updates the grid cells with the given filled cells.
        /// </summary>
        /// <param name="filledCells">The filled cells.</param>
        public void UpdateGridCells(bool[,] filledCells)
        {
            Assert.IsNotNull(cells, "The grid must be initialized before updating cells.");
            Assert.IsTrue(filledCells.GetLength(0) == gridSize.x && filledCells.GetLength(1) == gridSize.y, 
                "The grid size must match the size of the filled cells array.");

            for (int x = 0; x < gridSize.x; ++x)
            {
                for (int y = 0; y < gridSize.y; ++y)
                {
                    Image image = cells[x, y].GetComponent<Image>();
                    image.color = filledCells[x, y] ? filledColor : emptyColor;
                }
            }
        }

        #endregion

        #region Helpers

        private void DestroyGrid()
        {
            if (cells == null)
            {
                return;
            }

            foreach (var cell in cells)
            {
                if (cell != null)
                {
                    Destroy(cell);
                }
            }
        }

        private void CreateEmptyGrid(Vector2Int gridSize)
        {
            // Create new cells.
            cells = new GameObject[gridSize.x, gridSize.y];
            for (int x = 0; x < gridSize.x; ++x)
            {
                for (int y = 0; y < gridSize.y; ++y)
                {
                    CreateCell(x, y, false);
                }
            }
        }

        private void CreateCell(int x, int y, bool filled)
        {
            var cell = Instantiate(gridCell, transform);
            cell.name = $"Cell ({x}, {y})";
            cell.transform.localPosition = new Vector3(x * (cellSize.x + cellSpacing.x), y * (cellSize.y + cellSpacing.y), 0);
            cell.transform.localScale = Vector3.one;
            RectTransform rectTransform = cell.GetComponent<RectTransform>();
            Assert.IsNotNull(rectTransform, "The grid cell prefab must have a RectTransform component.");
            rectTransform.sizeDelta = cellSize;
            Image image = cell.GetComponent<Image>();
            Assert.IsNotNull(image, "The grid cell prefab must have an Image component.");
            image.color = filled ? filledColor : emptyColor;
            cells[x, y] = cell;
        }

        private void CenterGrid()
        {
            if (cells == null)
            {
                return;
            }

            Vector2 gridSizePixels = new Vector2(gridSize.x * (cellSize.x + cellSpacing.x), gridSize.y * (cellSize.y + cellSpacing.y));
            Vector2 gridPosition = new Vector2(-gridSizePixels.x / 2.0f, -gridSizePixels.y / 2.0f);
            transform.localPosition = initialPosition + gridPosition;
        }

        #endregion
    }
}
