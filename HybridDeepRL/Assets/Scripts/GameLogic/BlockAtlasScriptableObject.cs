using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace GameLogic
{
	[CreateAssetMenu(fileName = "BlockAtlas", menuName = "Gamelogic/BlockAtlas", order = 2)]
	public class BlockAtlasScriptableObject : ScriptableObject
    {
        public List<BlockScriptableObject> m_blocks;

        public int getAtlasSize(){
            return m_blocks.Count;
        }
        public BlockScriptableObject getBlock(int index){
            string log = string.Format("atlas size {0}, index {1}", m_blocks.Count, index);
            Debug.Log(log);
            Debug.Assert(0 <= index && index < m_blocks.Count);
            return m_blocks[index];
        }
    }
}
